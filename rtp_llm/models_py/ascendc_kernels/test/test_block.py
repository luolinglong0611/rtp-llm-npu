import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

BLOCK_PATH = Path(__file__).resolve().parents[1] / "block.py"


def _source_tree():
    return ast.parse(BLOCK_PATH.read_text(encoding="utf-8"), filename=str(BLOCK_PATH))


class BlockStaticTest(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_source_tree(), str(BLOCK_PATH), "exec")

    def test_triton_fallbacks_are_lazy(self):
        eager_imports = [
            node
            for node in _source_tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported_modules = []
        for node in eager_imports:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            else:
                imported_modules.append(node.module or "")

        self.assertFalse(
            any("triton" in module for module in imported_modules), imported_modules
        )

    def test_public_signatures_match_triton_implementation(self):
        function_names = {
            "load_initial_state_from_block_map",
            "store_ssm_state_to_block_map",
        }
        actual = {
            node.name: ast.dump(node.args, include_attributes=False)
            for node in _source_tree().body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        }
        triton_tree = ast.parse(
            """
def load_initial_state_from_block_map(
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    conv_states: torch.Tensor,
    initial_states: torch.Tensor,
    seq_size_per_block: int,
    block_v: int = 64,
):
    pass

def store_ssm_state_to_block_map(
    h: torch.Tensor,
    final_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_map: torch.Tensor,
    ssm_states: torch.Tensor,
    seq_size_per_block: int,
    chunk_size: int,
    block_v: int = 64,
):
    pass
"""
        )
        expected = {
            node.name: ast.dump(node.args, include_attributes=False)
            for node in triton_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        }
        self.assertEqual(actual, expected)

    def test_npu_dispatch_accepts_both_device_names(self):
        tree = _source_tree()
        dispatch_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_NPU_DEVICE_TYPES"
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "_is_npu_device")
        ]
        namespace = {"torch": SimpleNamespace(device=object)}
        dispatch_module = ast.fix_missing_locations(
            ast.Module(body=dispatch_nodes, type_ignores=[])
        )
        exec(compile(dispatch_module, str(BLOCK_PATH), "exec"), namespace)
        is_npu_device = namespace["_is_npu_device"]

        self.assertTrue(is_npu_device(SimpleNamespace(type="npu")))
        self.assertTrue(is_npu_device(SimpleNamespace(type="privateuseone")))
        self.assertFalse(is_npu_device(SimpleNamespace(type="cuda")))
        self.assertFalse(is_npu_device(SimpleNamespace(type="cpu")))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class BlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("ascendc_block_test", BLOCK_PATH)
        cls.block = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.block)

    def test_load_initial_state_from_paged_block_map(self):
        state_elements = 2 * 2 * 3
        conv_storage = torch.arange(
            6 * (state_elements + 3), dtype=torch.bfloat16
        ).reshape(6, state_elements + 3)
        conv_states = conv_storage[:, :state_elements].view(6, 2, 2, 3)
        block_map = torch.tensor([[5, 2, 4], [3, 1, 0]], dtype=torch.int32)
        prefix_lengths = torch.tensor([0, 5], dtype=torch.int32)
        initial_states = torch.full((2, 2, 3, 2), -1.0, dtype=torch.float32)
        storage_pointer = initial_states.data_ptr()

        with mock.patch.object(self.block, "_is_npu_device", return_value=True):
            result = self.block.load_initial_state_from_block_map(
                prefix_lengths,
                block_map,
                conv_states,
                initial_states,
                seq_size_per_block=4,
            )

        self.assertIsNone(result)
        self.assertEqual(initial_states.data_ptr(), storage_pointer)
        torch.testing.assert_close(
            initial_states[0], torch.zeros_like(initial_states[0])
        )
        torch.testing.assert_close(
            initial_states[1], conv_states[1].transpose(-1, -2).float()
        )

    def test_store_final_and_global_intermediate_states(self):
        # Both sequences have three chunks.  Their aligned middle-block states
        # are h[2] and h[5], demonstrating that h uses a global chunk ordinal.
        h_flat = torch.arange(6 * 1 * 2 * 3, dtype=torch.float32).reshape(6, 1, 2, 3)
        final_states = (
            torch.arange(2 * 1 * 2 * 3, dtype=torch.float32).reshape(2, 1, 2, 3) + 100
        )
        prefix_lengths = torch.tensor([0, 4], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 6, 11], dtype=torch.int32)
        block_map = torch.tensor([[1, 2, 6], [3, 4, 5]], dtype=torch.int32)

        for h in (h_flat, h_flat.unsqueeze(0)):
            with self.subTest(h_shape=h.shape):
                cache_storage = torch.full((7, 9), -1.0, dtype=torch.bfloat16)
                ssm_states = cache_storage[:, :6].view(7, 1, 3, 2)
                storage_pointer = ssm_states.data_ptr()

                with mock.patch.object(self.block, "_is_npu_device", return_value=True):
                    result = self.block.store_ssm_state_to_block_map(
                        h,
                        final_states,
                        prefix_lengths,
                        cu_seqlens,
                        block_map,
                        ssm_states,
                        seq_size_per_block=4,
                        chunk_size=2,
                    )

                self.assertIsNone(result)
                self.assertEqual(ssm_states.data_ptr(), storage_pointer)
                torch.testing.assert_close(
                    ssm_states[1], h_flat[2].transpose(-1, -2).bfloat16()
                )
                torch.testing.assert_close(
                    ssm_states[2], final_states[0].transpose(-1, -2).bfloat16()
                )
                torch.testing.assert_close(
                    ssm_states[4], h_flat[5].transpose(-1, -2).bfloat16()
                )
                torch.testing.assert_close(
                    ssm_states[5], final_states[1].transpose(-1, -2).bfloat16()
                )
                for untouched_block in (0, 3, 6):
                    torch.testing.assert_close(
                        ssm_states[untouched_block],
                        torch.full_like(ssm_states[untouched_block], -1.0),
                    )
                torch.testing.assert_close(
                    cache_storage[:, 6:],
                    torch.full_like(cache_storage[:, 6:], -1.0),
                )

    def test_store_skips_non_positive_physical_block(self):
        ssm_states = torch.full((1, 1, 1, 1), -1.0, dtype=torch.float32)

        with mock.patch.object(self.block, "_is_npu_device", return_value=True):
            self.block.store_ssm_state_to_block_map(
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.full((1, 1, 1, 1), 9.0, dtype=torch.float32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                ssm_states,
                seq_size_per_block=4,
                chunk_size=2,
            )

        torch.testing.assert_close(ssm_states, torch.full_like(ssm_states, -1.0))

    def test_store_requires_float32_sources(self):
        h = torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16)
        final_states = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

        with mock.patch.object(self.block, "_is_npu_device", return_value=True):
            with self.assertRaisesRegex(
                AssertionError, "h and final_states must be float32"
            ):
                self.block.store_ssm_state_to_block_map(
                    h,
                    final_states,
                    torch.tensor([0], dtype=torch.int32),
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([[1]], dtype=torch.int32),
                    torch.zeros((2, 1, 1, 1), dtype=torch.float32),
                    seq_size_per_block=4,
                    chunk_size=2,
                )


if __name__ == "__main__":
    unittest.main()
