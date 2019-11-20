from .generalized_trn import GeneralizedTRN

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
