import torch
import torch.nn.functional as F
import torch.nn as nn

class Proden_loss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output1,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss


class Proden_loss_RECORDS(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target
        self.feat_mean = None


    def forward(self,output1,feat,model,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        if self.feat_mean is None:
            self.feat_mean = 0.1*feat.detach().mean(0)
        else:
            self.feat_mean = 0.9*self.feat_mean + 0.1*feat.detach().mean(0)

        
        if update_target:
            bias = model.module.fc(self.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            logits = output1 - torch.log(bias + 1e-9) 
            output = F.softmax(logits, dim=1)


            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

class Proden_loss_prior(nn.Module):
    def __init__(self, target,partial_prior):
        super().__init__()
        self.confidence = target
        self.feat_mean = None
        self.partial_prior = torch.tensor(partial_prior).float().cuda()


    def forward(self,output1,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        
        if update_target:

            logits = output1 - torch.log(self.partial_prior + 1e-9) 
            output = F.softmax(logits, dim=1)


            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output_w,output_s,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(1-pred_w)-torch.log(1-pred_s))
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        loss = sup_loss1 + con_loss
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w
            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss


class CORR_loss(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output_w,output_s,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        loss = sup_loss1 + con_loss
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w
            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

class CORR_loss_RECORDS(nn.Module):
    def __init__(self, target, m = 0.9):
        super().__init__()
        self.confidence = target
        self.init_confidence = target.clone()
        self.feat_mean = None
        self.m = m


    def forward(self,output_w,output_s,feat_w,feat_s,model,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        if torch.any(torch.isnan(sup_loss)):
            print("sup_loss:nan")
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        if torch.any(torch.isnan(con_loss)):
            print("con_loss:nan")
        loss = sup_loss1 + con_loss

        if self.feat_mean is None:
            self.feat_mean = (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        else:
            self.feat_mean = self.m*self.feat_mean + (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)

        
        if update_target:
            bias = model.module.fc(self.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            logits_s = output_s - torch.log(bias + 1e-9) 
            logits_w = output_w - torch.log(bias + 1e-9) 
            pred_s = F.softmax(logits_s, dim=1)
            pred_w = F.softmax(logits_w, dim=1)


            # revisedY = target.clone()
            revisedY = self.init_confidence[index,:].clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w            
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY
            # new_target = torch.where(torch.isnan(new_target), self.init_confidence[index,:].clone(), new_target)

            self.confidence[index,:]=new_target.detach()

        return loss

class CORR_loss_RECORDS_mixup(nn.Module):
    def __init__(self, target, m = 0.9, mixup=0.5):
        super().__init__()
        self.confidence = target
        self.init_confidence = target.clone()
        self.feat_mean = None
        self.m = m
        self.mixup = mixup


    def forward(self,output_w,output_s,output_w_mix,output_s_mix,feat_w,feat_s,model,index,pseudo_label_mix,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        if torch.any(torch.isnan(sup_loss)):
            print("sup_loss:nan")
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        if update_target:
            pred_s_mix = F.softmax(output_s_mix, dim=1)
            pred_w_mix = F.softmax(output_w_mix, dim=1)
            neg2 = (pseudo_label_mix==0).float()
            sup_loss2 = neg2 * (-torch.log(abs(1-pred_w_mix)+1e-9)-torch.log(abs(1-pred_s_mix)+1e-9))
            sup_loss_2 = torch.sum(sup_loss2) / sup_loss2.size(0)
            con_loss_2 = F.kl_div(torch.log_softmax(output_w_mix,dim=1),pseudo_label_mix,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s_mix,dim=1),pseudo_label_mix,reduction='batchmean')
        else:
            sup_loss_2 = 0
            con_loss_2 = 0

        if torch.any(torch.isnan(con_loss)):
            print("con_loss:nan")
        loss = sup_loss1 + con_loss + self.mixup * (sup_loss_2+ con_loss_2)

        if self.feat_mean is None:
            self.feat_mean = (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        else:
            self.feat_mean = self.m*self.feat_mean + (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        
        
        if update_target:
            bias = model.module.fc(self.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            logits_s = output_s - torch.log(bias + 1e-9) 
            logits_w = output_w - torch.log(bias + 1e-9) 
            pred_s = F.softmax(logits_s, dim=1)
            pred_w = F.softmax(logits_w, dim=1)


            # revisedY = target.clone()
            revisedY = self.init_confidence[index,:].clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w            
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

