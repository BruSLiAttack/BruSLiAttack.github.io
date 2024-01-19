import torch
import numpy as np
from utils import *

# main attack
class BruSLeAttack():
    def __init__(self,
                model,
                n = 4,  # a number of perturb pixels
                pop_size = 10,
                lamda=0.01,
                m1=0.24,
                m2=0.997,
                seed = None,
                flag=True,
                ftype='ce'):

        self.model = model
        self.n_pix = n 
        self.pop_size = pop_size
        self.lamda = lamda
        self.m1 = m1
        self.m2 = m2
        self.seed = seed
        self.flag = flag
        self.ftype = ftype

    def selection(self,x1,f1,x2,f2):

        # x1, f1: parent
        # x2, f2: offspring
        # failure approach: update when offspring worst
        xo = x1.copy()
        fo = f1
        fh_update = 1 # update for Failure approach
        if f2<f1:
            fo = f2
            xo = x2
            fh_update = 0 # non_update => but could be used to update for Success approach

        return xo,fo, fh_update

    
    def power_stepdecay_scheduler(self,q):
        
        lamda = self.lamda * (pow(q + 1.0, -self.m1) + self.m2**(q+1))/2
        
        return lamda

    def convert1D_to_2D(self,idx,wi):
        c1 = idx //wi
        c2 = idx % wi # = idx - c1 * wi
        return c1, c2
        
    def modify(self,pop,oimg,timg):
        wi = oimg.shape[2]
        img = oimg.clone()
        p = np.where(pop == 1)
        c1,c2 = self.convert1D_to_2D(p[0],wi)
        img[:,:,c1,c2] = timg[:,:,c1,c2]

        return img

    def rand_init(self,oimg,timg,olabel,tlabel):

        nqry = 0
        wi = oimg.shape[2]
        he = oimg.shape[3]
        p = np.zeros(wi*he).astype(int) # 1 => perturbed (starting); 0 => non-perturbed (ori)
        feval = torch.zeros(self.pop_size).cuda() #+ f_score_eval(model,ori,tar,ori_label,tar_label,p,flag,ftype)
        pop = []
            
        for i in range(self.pop_size):
            p = np.zeros(wi*he).astype(int)
            idx = np.random.choice(wi*he, self.n_pix, replace = False)
            p[idx] = 1
            nqry += 1
            fitness,_ = self.feval_score(oimg,timg,olabel,tlabel,p)
            pop.append(p)
            feval[i] = fitness
                    
        return pop,nqry,feval

    def feval_score(self,oimg,timg,olabel,tlabel,pop):

        xp = self.modify(pop,oimg,timg)
        pred_score = self.model(xp)

        rank = np.argsort(-pred_score[0].detach().cpu().numpy()) 
        top1_id = rank[0].item()
        top2_id = rank[1].item()
        
        with torch.no_grad():
            if self.flag == True:
                tscore = pred_score[0,tlabel].item()                
                top_score = pred_score[0,top1_id].item()
                outp_margin = top_score - tscore # successful if margin < 0
                if self.ftype=='margin':
                    outp = outp_margin
                elif self.ftype=='ce': # = nn.CrossEntropyLoss(): inlcude log_softmax
                    y = torch.tensor([tlabel]).cuda()
                    outp = F.cross_entropy(pred_score, y, reduction='none')#.detach()

            else:
                pred_score = F.softmax(pred_score,dim = 1)
                oscore = pred_score[0,olabel].item()
                top_score = pred_score[0,top2_id].item() if top2_id!= olabel else pred_score[0,top1_id].item()
                outp_margin = oscore - top_score # successful if margin < 0        
                outp = outp_margin

        return outp,outp_margin

# =================================== Method 3 =======================================
    def visited_pixel_map(self,visit_map,p): # record history of all pixels (search space). it can be use to determine age of pixel
        out_a = visit_map.copy()
        out_a += p # number of time of visit
        out_b = np.clip(out_a,0,1) # count visited or not
        n_px = out_b.sum()
        return out_a,out_b,n_px


# =================================== Method 4 =======================================
    def sampling(self,fail_map,visit_map,bias_map,p,m):

        ep = 1e-2# 1e-3
        num = fail_map + ep# s+a: number of succ + a
        den = visit_map + ep# s+a + N-s+b = N+a+b: number of Visit + a+b 
        # look at book "ML Fundamental - VHTiep" about smoothing term used in update (Naive Bayes Net/MAP=MLE)
        # it is a need to avoid den = 0 and num = 0
        # num/den < 1
        
        # 1. select remaing bits
        mask = num/den*p 
        #---------------------------------------------------
        idxs = np.where(mask>0)[0] # => pixel position
        prob = mask[idxs] # => value of 'fail_pix' matrix
        prob = prob/prob.sum()    
        outp = np.zeros(*p.shape).astype(int)
        n_p = int(p.sum()*m)
        if n_p<1:
            n_p=1
        idx = np.random.choice(idxs,p.sum()-n_p,p=prob,replace=False)
        outp[idx]=1
        #---------------------------------------------------

        tmp = np.logical_xor(outp,p)
        idx = np.where(tmp==1)[0]
        old = idx.copy()

        # 2. select new bits to add in
        mask = (num/den * (bias_map))*(1-p)
        idxs = np.where(mask>0)[0]
        prob = mask[idxs] # => value of 'fail_pix' matr
        prob = prob/prob.sum()    
        idx = np.random.choice(idxs,n_p,p=prob,replace=False)
        outp[idx]=1

        return outp,old

    def perturb(self,oimg,timg,olabel,tlabel,max_query=10000):

        terminate  = False

        # 1a. initialize
        pop, nqry,feval = self.rand_init(oimg,timg,olabel,tlabel)
        visit_map = np.zeros(len(pop[0]))
        for i in range(self.pop_size):
            visit_map,_,_= self.visited_pixel_map(visit_map,pop[i])
        fail_map = np.zeros(len(pop[0])) 
        bias_map = torch.abs(oimg-timg)[0].sum(axis=0).reshape(-1).cpu().numpy()/3

        # 1b. find the best
        rank = torch.argsort(feval) 
        best_idx = rank[0].item()
        p = pop[best_idx]
        fp = feval[best_idx]
        
        # 2. evolution

        while not terminate:

            # a. sampling
            lamda = self.power_stepdecay_scheduler(nqry)
            offspring,old = self.sampling(fail_map,visit_map,bias_map, p,lamda)

            # d. update 
            nqry += 1
            ftemp,f_mg = self.feval_score(oimg,timg,olabel,tlabel,offspring)
                
            # selection
            p,fp,fh_update = self.selection(p,fp,offspring,ftemp)
            
            # -------- Update -----------

            if fh_update: #
                fail_map[old] += 1
            visit_map[old] += 1 
            visit_map+=offspring

            # ----------------------------

            # adv = self.modify(p,oimg,timg)
            adv = self.modify(offspring,oimg,timg)
        
            if f_mg>0:
                if nqry%500 == 0:
                    print('len(xbest): %d; fbest: %2.3f; nqry: %d; L0: %d; n pix: %d; margin score: %2.3f' 
                      %(np.sum(p),fp, nqry,l0b(adv,oimg),self.n_pix*lamda,f_mg))
                if nqry > max_query:
                    print(f'attack terminate due to over query limit!')
                    terminate = True
            else:
                alabel = self.model.predict_label(adv).item()
                print(f'attack successful at {nqry}, adv label: {alabel}; tlabel={tlabel}; f_mg={f_mg}')
                terminate = True        

        dist = l0b(adv,oimg)

        return adv, nqry, dist