"""Resume complete method state as well as HF optimizer, sampler, and RNG state."""
import codecs
from pathlib import Path
import numpy as np
import torch
from .adapters import restore_adapter, save_adapter


class MethodCheckpointMixin:
    def _save(self, output_dir=None, state_dict=None):
        if self._dlz_lora:
            save_adapter(self.model, output_dir or self.args.output_dir, self.processing_class)
        else:
            return super()._save(output_dir, state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self._dlz_lora:
            restore_adapter(model or self.model, resume_from_checkpoint)
        else:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self):
        if self._dlz_lora:
            restore_adapter(self.model, self.state.best_model_checkpoint)
        else:
            return super()._load_best_model()

    def _save_rng_state(self, output_dir):
        super()._save_rng_state(output_dir)
        state = {'method': self._dlz_method, 'skipped_steps': getattr(self, '_skipped_zo_steps', 0),
                 'last_finite_loss': getattr(self, '_last_finite_loss', None)}
        if hasattr(self, '_py_rng'):
            state['perturbation_rng'] = self._py_rng.getstate()
        if hasattr(self, '_perturbation_rng'):
            algorithm, keys, position, has_gauss, cached = self._perturbation_rng.get_state()
            state['numpy_perturbation_rng'] = (algorithm, keys.tolist(), position, has_gauss, cached)
        if hasattr(self, 'sparse_param_groups'):
            state['sparse'] = {
                g.name: {'threshold': None if g.threshold is None else g.threshold.detach().cpu(),
                         'shape': tuple(g.param.shape), 'initial_selected': g.initial_selected}
                for g in self.sparse_param_groups}
        torch.save(state, Path(output_dir)/'method_state.pt')

    def _load_rng_state(self, checkpoint):
        # Allow NumPy RNG array types when loading saved Trainer state.
        prior = list(torch.serialization.get_safe_globals())
        torch.serialization.add_safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('uint32')), codecs.encode,
        ])
        try:
            result = super()._load_rng_state(checkpoint)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(prior)
        if checkpoint is None:
            return result
        state = torch.load(Path(checkpoint)/'method_state.pt', map_location='cpu', weights_only=True)
        if state['method'] != self._dlz_method:
            raise ValueError('Checkpoint optimization method differs')
        if 'numpy_perturbation_rng' in state:
            algorithm, keys, position, has_gauss, cached = state['numpy_perturbation_rng']
            self._perturbation_rng = np.random.RandomState()
            self._perturbation_rng.set_state((algorithm, np.array(keys, dtype=np.uint32),
                                           position, has_gauss, cached))
        if hasattr(self, '_py_rng'):
            self._py_rng.setstate(state['perturbation_rng'])
            self._skipped_zo_steps = state['skipped_steps']
            loss = state['last_finite_loss']
            self._last_finite_loss = None if loss is None else loss.to(self.args.device)
        if hasattr(self, 'sparse_param_groups'):
            saved = state['sparse']
            if saved.keys() != {g.name for g in self.sparse_param_groups}:
                raise ValueError('Sparse checkpoint parameter set differs')
            for g in self.sparse_param_groups:
                item = saved[g.name]
                if tuple(g.param.shape) != item['shape']:
                    raise ValueError('Sparse checkpoint shape differs')
                threshold = item['threshold']
                g.threshold = None if threshold is None else threshold.to(device=g.param.device, dtype=g.param.dtype)
                g.initial_selected = item['initial_selected']
            self.sparse_initial_selected_entries = sum(g.initial_selected for g in self.sparse_param_groups)
            self.trainable_dim = self.sparse_initial_selected_entries
        return result
