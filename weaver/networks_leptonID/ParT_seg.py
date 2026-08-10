import os
import torch
from weaver.utils.logger import _logger
from weaver.nn.model.ParticleTransformer import ParticleTransformer, Embed


class ParticleTransformerSeg(torch.nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_type='pp',
                 pair_input_dim=None,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=(128, 512, 128),
                 pair_embed_dims=(64, 64, 64),
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params=None,
                 fc_params=(),
                 activation='gelu',
                 # misc
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
                                        pair_input_type=pair_input_type,
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        # misc
                                        trim=False,
                                        for_segmentation=True,
                                        for_inference=for_inference,
                                        use_amp=use_amp)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (batch, seq_len, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=1)

            return self.part(x, v, mask)


def get_model(data_config, **kwargs):

    def _make_cfg(new_cfg):
        _cfg = dict(
            activation='swiglu',
            scale_attn_mask=False,
            scale_fc=False, scale_attn=False, scale_heads=False, scale_resids=False,
        )
        if new_cfg is not None:
            _cfg.update(new_cfg)
        return _cfg
    block_params = _make_cfg(kwargs.pop('block_params', None))

    cfg = dict(
        pf_input_dim=len(data_config.input_dicts['jet_features']),
        sv_input_dim=len(data_config.input_dicts['lep_features']),
        num_classes=len(data_config.labels['names']),  # !
        # network configurations
        pair_input_type='pp',
        pair_input_dim=None,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        block_params=block_params,
        # cls blocks
        num_cls_layers=0,  # !
        cls_block_params=None,
        fc_params=(),
        activation='gelu',
        # misc
        trim=False,  # !
        for_inference=False,
    )

    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerSeg(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((2,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
