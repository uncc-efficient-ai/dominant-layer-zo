"""Orchestrate the preserved numerical backends with portable configuration."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import TrainingArguments, set_seed
from .config import signature
from .models import backend, load_model
from .data import make_data
from .adapters import attach_adapter
from .checkpointing import MethodCheckpointMixin
from .runtime import atomic_json, environment, run_lock, implementation_signature


def make_trainer(c, model, tokenizer, train, dev, collator):
    t, k, f = c['training'], c['checkpoint'], c['fo']
    is_fo = t['method'] in ('fo', 'fo_lora')
    is_lora = t['method'].endswith('_lora')
    if t['method'] in ('agzo', 'sparse_mezo'):
        if t['scope'] != 'full' or t['layer'] is not None or t['part'] != 'layer':
            raise ValueError('SparseMeZO and AGZO require full scope with layer=null and part=layer')
        if t['noise'] != 'gaussian' or t['directions'] != 1 or t['max_zo_coeff'] is not None:
            raise ValueError('SparseMeZO and AGZO require one unclipped Gaussian direction')
        model.config.fp32_attention_scores = True
        if c['model']['family'] == 'mistral':
            model.config._attn_implementation = 'dlzo_mistral_fp32'
    if not is_fo and any(p.dtype == torch.bfloat16 for p in model.parameters()):
        raise ValueError('Raw bf16 ZO is unsupported; use float16 or float32')
    if next(model.parameters()).device.type == 'cuda' and torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES per training process')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one process and one visible GPU per experiment')
    if t['scope'] == 'layer' and t['layer'] >= len(model.model.layers):
        raise ValueError('Selected layer is outside the model')
    if is_lora:
        attach_adapter(c, model)
    args = TrainingArguments(
        output_dir=c['output_dir'], max_steps=t['steps'] if t['steps'] is not None else -1,
        num_train_epochs=f['epochs'] if is_fo else 1.,
        per_device_train_batch_size=t['batch_size'], per_device_eval_batch_size=t['batch_size'],
        gradient_accumulation_steps=f['gradient_accumulation_steps'] if is_fo else 1,
        gradient_checkpointing=f['gradient_checkpointing'] if is_fo else False,
        gradient_checkpointing_kwargs={'use_reentrant': False} if is_fo and f['gradient_checkpointing'] else None,
        learning_rate=t['lr'] if is_fo else 0., weight_decay=f['weight_decay'] if is_fo else 0.,
        max_grad_norm=f['max_grad_norm'] if is_fo else 0.,
        lr_scheduler_type=f['scheduler'] if is_fo else 'constant',
        warmup_ratio=f['warmup_ratio'] if is_fo else 0.,
        adam_beta1=f['adam_beta1'], adam_beta2=f['adam_beta2'], adam_epsilon=f['adam_epsilon'],
        optim='adamw_torch', fp16=is_fo and f['amp']=='fp16', bf16=is_fo and f['amp']=='bf16',
        eval_strategy='steps', eval_steps=t['eval_steps'], logging_steps=t['logging_steps'],
        save_strategy=k['strategy'], save_steps=k['save_steps'], save_total_limit=k['save_total_limit'],
        load_best_model_at_end=k['load_best_model_at_end'], metric_for_best_model='eval_loss',
        greater_is_better=False, remove_unused_columns=False, report_to=[],
        seed=t['seed'] if is_fo else c['data']['seed'], dataloader_num_workers=0,
        use_cpu=next(model.parameters()).device.type == 'cpu',
    )
    zo = dict(eps=t['epsilon'], lr=t['lr'], directions=t['directions'], layer_idx=t['layer'],
              seed=t['seed'], noise=t['noise'], part=t['part'], cast_trainable_fp32=False,
              max_zo_coeff=t['max_zo_coeff'] if is_lora else ('none' if t['max_zo_coeff'] is None else t['max_zo_coeff']),
              log_zo_internals=True, non_diff=t['non_diff'], freeze_layers=[],
              forward_autocast=t['forward_autocast'], perturb_math_fp32=c['lora']['perturb_math_fp32'],
              agzo=c['agzo'],
              sparse_selected_ratio=c['sparse']['selected_ratio'],
              upcast_loss_logits_fp32=t['upcast_loss_logits_fp32'],
              upcast_zo_scalar_fp32=t['upcast_zo_scalar_fp32'],
              non_diff_max_new_tokens=c['evaluation']['max_new_tokens'])
    def namespace(value):
        return SimpleNamespace(**{k:namespace(v) for k,v in value.items()}) if isinstance(value,dict) else value
    context = namespace({'task': {'tune': {'zo':zo, 'fo':zo, 'tune_dataset':{'name':c['data']['task']+'_mezo'}}},
                         'model': {'peft': {'lora': {'attach_scope':t['scope'], 'layer_idx':t['layer']}}}})
    if t['method'] in ('agzo', 'sparse_mezo'):
        from .methods.agzo import AGZOTrainerMixin
        from .sparse import SparseMeZOTrainerMixin
        base = backend(c['model']['family'], 'zo_' + t['scope']).MyTrainer
        mixin = AGZOTrainerMixin if t['method'] == 'agzo' else SparseMeZOTrainerMixin
        cls = type('MethodTrainer', (mixin, base), {})
    else:
        cls = backend(c['model']['family'], t['method'] + '_' + t['scope']).MyTrainer
    class TaskTrainer(MethodCheckpointMixin, cls):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if not getattr(self, 'non_diff', False):
                return super().prediction_step(model,inputs,prediction_loss_only,ignore_keys)
            with torch.no_grad():
                loss = self.compute_nondiff_loss(model,self._prepare_inputs(inputs))
            return loss.detach(), None, None

    trainer = TaskTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train,
                          eval_dataset=dev, data_collator=collator, task_config=context)
    trainer._dlz_method = t['method']; trainer._dlz_lora = is_lora
    # All preserved loss functions return a batch mean. They do not consume HF's
    # num_items_in_batch normalization; let Trainer scale gradient accumulation.
    if is_fo:
        trainer.model_accepts_loss_kwargs = False
        trainer.trainable_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainer.trainable_dim:
        raise ValueError('The requested method/scope selects no trainable coordinates')
    return trainer


def evaluate_model(c, model, tokenizer):
    e = c['evaluation']
    # Record the fp16 model cast performed by setup_eval.
    model.to(dtype=getattr(torch, e['dtype']))
    return backend(c['model']['family'],'evaluation').run_downstream_eval(
        model, tokenizer, tasks=[c['data']['task']+'_mezo'], max_length=e['max_length'],
        batch_size=e['batch_size'], max_samples=e['max_samples'],
        max_new_tokens=e['max_new_tokens'], seed=e['seed'])


def train(c, device='cuda', resume=None):
    with run_lock(c['output_dir'], resume=bool(resume)) as out:
        previous = None
        software = environment()
        code_signature = implementation_signature()
        if resume:
            previous = json.loads((out/'run.json').read_text())
            if previous['signature'] != signature(c):
                raise ValueError('Resume configuration does not match the saved run')
            if previous.get('implementation_signature') != code_signature:
                raise ValueError('Implementation changed; use a new run instead of resuming')
            if previous['environment']['packages'] != software['packages']:
                raise ValueError('Software versions changed; refusing to resume')
            if previous.get('status') == 'complete':
                raise ValueError('This run is complete; use a new output_dir')
            cp = Path(resume).resolve()
            if cp.parent != out.resolve():
                raise ValueError('Resume checkpoint must belong to this output_dir')
            for name in ['trainer_state.json','rng_state.pth','method_state.pt','optimizer.pt','scheduler.pt']:
                if not (cp/name).is_file():
                    raise ValueError(f'Incomplete resume checkpoint: missing {name}')
        record = {**(previous or {}), 'signature':signature(c),'config':c,'status':'running','environment':software,
                  'implementation_signature':code_signature}
        atomic_json(out/'run.json', record)
        atomic_json(out/'config.json', c)
        try:
            set_seed(c['data']['seed'])
            model, tokenizer = load_model(c['model'],device)
            train_data, dev, collator, data_info = make_data(c,tokenizer)
            if previous and previous.get('data') != data_info:
                raise ValueError('Dataset fingerprint or split changed; refusing to resume')
            record['data'] = data_info
            model_revision = getattr(model.config, '_commit_hash', None)
            if previous and previous.get('model_revision') != model_revision:
                raise ValueError('Resolved model revision changed; refusing to resume')
            record['model_revision'] = model_revision
            atomic_json(out/'run.json',record)
            trainer = make_trainer(c,model,tokenizer,train_data,dev,collator)
            if hasattr(trainer, 'sparse_param_groups'):
                from .sparse import sparse_metadata
                record['sparsity'] = sparse_metadata(trainer)
                atomic_json(out/'run.json', record)
            if c['training']['method'] == 'agzo':
                record['agzo'] = {**c['agzo'], 'reference': 'arxiv:2601.17261v1',
                                  'basis_padding': 'included',
                                  'code_reference': '623e09a5a7dfdee697a603ec1e7aa01e3f5e6290',
                                  'rank_scaling': 'inverse_sqrt_rank', 'estimator': 'central',
                                  'restoration': 'exact_cpu_snapshot', 'update_dtype': 'float32',
                                  'cpu_snapshot_bytes': sum(p.numel() * p.element_size()
                                      for p in model.parameters() if p.requires_grad)}
                atomic_json(out/'run.json', record)
                if next(model.parameters()).device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
            train_output = trainer.train(resume_from_checkpoint=resume)
            if c['training']['method'] == 'agzo':
                forwards = 3
                record['agzo'].update(
                    training_objective_evaluations=trainer.state.global_step * forwards,
                    basis_refreshes=trainer.state.global_step,
                    invocation_training_seconds=train_output.metrics['train_runtime'],
                    invocation_peak_allocated_bytes=(
                        torch.cuda.max_memory_allocated()
                        if next(model.parameters()).device.type == 'cuda' else None))
            validation = trainer.evaluate()
            if not math.isfinite(validation['eval_loss']):
                raise FloatingPointError('Non-finite internal dev loss')
            skipped = getattr(trainer, '_skipped_zo_steps', 0)
            if skipped and skipped >= trainer.state.global_step:
                raise FloatingPointError('All ZO steps were skipped')
            selection = {'metric':'internal_dev_loss','value':validation['eval_loss'],
                         'split':'held_out_train','step':trainer.state.global_step,
                         'best_checkpoint':trainer.state.best_model_checkpoint,
                         'best_dev_loss':trainer.state.best_metric}
            atomic_json(out/'validation.json',selection)
            # Save before final evaluation changes precision.
            if c['checkpoint']['save_final']:
                trainer.save_model(str(out/'final'))
                tokenizer.save_pretrained(out/'final')
            trainer.save_state()
            metrics = None
            if c['evaluation']['after_training']:
                metrics = evaluate_model(c,model,tokenizer)
                atomic_json(out/'downstream_eval.json',metrics)
            record.update(status='complete',steps=trainer.state.global_step,
                          skipped_zo_steps=skipped,metrics=metrics,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          optimization_coordinates=trainer.trainable_dim,
                          sparse_initial_selected_entries=getattr(trainer, 'sparse_initial_selected_entries', None),
                          adapter=getattr(model, '_dlz_adapter_spec', None))
            atomic_json(out/'run.json',record)
        except BaseException as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            atomic_json(out/'run.json',record)
            raise
