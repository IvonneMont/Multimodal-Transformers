import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class FastClf(nn.Module):
    def __init__(self, args):
        super(FastClf, self).__init__()
        if args.model=="FastImage":
            self.input_sz=args.num_features_img*args.size_sequence_img
        else:
            self.input_sz=args.input_sz_fast
        self.dropout=nn.Dropout(p=0.5)
        self.clf = nn.Linear(self.input_sz, args.n_classes)

    def forward(self, x):
        x=torch.reshape(x,(x.size()[0],self.input_sz))
        x=self.dropout(x)
        out = self.clf(x)
        return out
    
class MultFusion(nn.Module):
    def __init__(self,args):
        super(MultFusion, self).__init__()
        self.args = args
        args.model="FastImage"
        self.clf_image=FastClf(args)
        args.model="MultFusion"
        self.clf_text=FastClf(args)
        self.Wt = Parameter(torch.randn(13).type(torch.FloatTensor), requires_grad=True).cuda()
        self.Wi = Parameter(torch.randn(13).type(torch.FloatTensor), requires_grad=True).cuda()
        
    def forward(self, txt, img):
        phi_t=self.clf_text(txt)
        phi_p=self.clf_image(img)
        alpha_t=F.softmax(self.Wt)
        alpha_p=F.softmax(self.Wi)
        out = alpha_t*phi_t+alpha_p*phi_p
        return out