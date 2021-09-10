from models.bert import BertClf
from models.mmbt import MultimodalBertClf
from models.mult import MULTModel
from models.mult2 import MULTModel2

MODELS = {
    "bert": BertClf,
    "mmbt": MultimodalBertClf,
    "mult": MULTModel,
    "mult2": MULTModel2,
}


def get_model(args):
    return MODELS[args.model](args)