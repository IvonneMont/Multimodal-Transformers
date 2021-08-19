from models.bert import BertClf
from models.mmbt import MultimodalBertClf
from models.mult import MULTModel

MODELS = {
    "bert": BertClf,
    "mmbt": MultimodalBertClf,
    "mult": MULTModel,
}


def get_model(args):
    return MODELS[args.model](args)