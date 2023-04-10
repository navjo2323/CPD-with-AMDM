import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


def mnorm_iter(T,A,B,C):
    calibrate_norms(A,B,C)
    flag =0
    Bdag = la.pinv(B).T
    Cdag = la.pinv(C).T
    Anew = np.einsum("ijk,jr,kr->ir",T,Bdag,Cdag,optimize=True)
    #print("A change is",la.norm(A-Anew))
    #print("A nchange is",scalinvar_distance(A,Anew))
    A = Anew
    Adag = la.pinv(A).T
    Bnew = np.einsum("ijk,ir,kr->jr",T,Adag,Cdag,optimize=True)
    #print("B change is",la.norm(B-Bnew))
    #print("B nchange is",scalinvar_distance(B,Bnew))
    B = Bnew
    Bdag = la.pinv(B).T
    Cnew = np.einsum("ijk,ir,jr->kr",T,Adag,Bdag,optimize=True)
    #print("C change is",la.norm(C-Cnew))
    #print("C nchange is",scalinvar_distance(C,Cnew))
    if scalinvar_distance(C,Cnew) < 1e-14:
        flag =1
    C = Cnew
    return [A,B,C],flag

def orth_mnorm_iter(T,Q1,Q2,Q3):
    flag = 0
    U1,s1,Vt1 = la.svd(np.einsum('ijk,jr,kr->ir',T,Q2,Q3,optimize=True),full_matrices=False)
    Q1 = U1@Vt1
    U2,s2,Vt2 = la.svd(np.einsum('ijk,ir,kr->jr',T,Q1,Q3,optimize=True),full_matrices=False)
    Q2  = U2@Vt2
    U3,s3,Vt3 = la.svd(np.einsum('ijk,ir,jr->kr',T,Q1,Q2,optimize=True),full_matrices=False)
    Q3new = U3@Vt3
    if scalinvar_distance(Q3,Q3new) < 1e-15:
        flag =1
    Q3 = Q3new.copy()
    return [U1,U2,U3],[Vt1,Vt2,Vt3],[s1,s2,s3],flag

def gen_mnorm_iter(T,A,B,C):
    flag = 0
    left1,sing1,right1 = la.svd(B,full_matrices=False)
    left2,sing2,right2 = la.svd(C,full_matrices=False)
    B = B/la.norm(B,axis=0)
    C = C/la.norm(C,axis=0)
    M = left1@np.diag(sing1/2)@left1.T
    cont_B = M@B
    M = left2@np.diag(sing2/2)@left2.T
    cont_C = M@C
    A = np.einsum('ijk,jr,kr->ir',T,cont_B,cont_C,optimize=True)
    left,sing,right = la.svd(A,full_matrices=False)
    #cont_A = left@np.diag(f(sing))@right
    A = A/la.norm(A,axis=0)
    M = left@np.diag(sing/2)@left.T
    cont_A = M@A
    B = np.einsum('ijk,ir,kr->jr',T,cont_A,cont_C,optimize=True)
    left,sing,right = la.svd(B,full_matrices=False)
    B = B/la.norm(B,axis=0)
    M = left@np.diag(sing/2)@left.T
    #cont_B = left@np.diag(f(sing))@right
    cont_B = M@B
    Cnew = np.einsum('ijk,ir,jr->kr',T,cont_A,cont_B,optimize=True)
    Cnew = Cnew/la.norm(Cnew,axis=0)
    if scalinvar_distance(C,Cnew) < 1e-14:
        flag =1
    C = Cnew.copy()
    return [A,B,C],flag



def orth_mnorm_iter2(T,Q1,Q2,Q3):
    flag = 0
    Q1,_ = la.qr(np.einsum('ijk,jr,kr->ir',T,Q2,Q3,optimize=True))
    Q2,_ = la.qr(np.einsum('ijk,ir,kr->jr',T,Q1,Q3,optimize=True))
    Q3new,_ = la.qr(np.einsum('ijk,ir,jr->kr',T,Q1,Q2,optimize=True))
    if scalinvar_distance(Q3,Q3new) < 1e-14:
        flag =1
    Q3 = Q3new.copy()
    return [Q1,Q2,Q3],flag

def orth_mnorm_iter_eig(T,Q1,Q2,Q3,up):
    flag = 0
    Q1,_ = la.qr(np.einsum('ijk,jr,kr->ir',T,Q2@up,Q3@up,optimize=True))
    Q2,_ = la.qr(np.einsum('ijk,ir,kr->jr',T,Q1@up,Q3@up,optimize=True))
    Q3new,_ = la.qr(np.einsum('ijk,ir,jr->kr',T,Q1@up,Q2@up,optimize=True))
    if scalinvar_distance(Q3,Q3new) < 1e-15:
        flag =1
    Q3 = Q3new.copy()
    return [Q1,Q2,Q3],flag

def orth_mnorm_iter3(T,Q1,Q2,Q3,Rs):
    flag = 0
    Q1,new_R1 = la.qr(np.einsum('ijk,jr,kr->ir',T,Q2@Rs[1],Q3@Rs[2],optimize=True))
    new_R1 = new_R1/la.norm(new_R1,axis=0)
    Q2,new_R2 = la.qr(np.einsum('ijk,ir,kr->jr',T,Q1@new_R1,Q3@Rs[2],optimize=True))
    new_R2= new_R2/la.norm(new_R2,axis=0)
    Q3new,new_R3 = la.qr(np.einsum('ijk,ir,jr->kr',T,Q1@new_R1,Q2@new_R2,optimize=True))
    if scalinvar_distance(Q3,Q3new) < 1e-13:
        flag =1
    Q3 = Q3new.copy()
    new_R3 = new_R3/la.norm(new_R3,axis=0)
    return [Q1,Q2,Q3],[new_R1,new_R2,new_R3],flag


def als_iter(Z,A,B,C):
    flag = 0
    BCdag = la.pinv(sla.khatri_rao(B,C)).T.reshape((Z.shape[1],Z.shape[2],A.shape[1]))
    Anew = np.einsum("ijk,jkr->ir",Z,BCdag)
    #print("A change is",la.norm(A-Anew))
    #print("A nchange is",scalinvar_distance(A,Anew))
    A = Anew
    ACdag = la.pinv(sla.khatri_rao(A,C)).T.reshape((Z.shape[0],Z.shape[2],A.shape[1]))
    Bnew = np.einsum("ijk,ikr->jr",Z,ACdag)
    #print("B change is",la.norm(B-Bnew))
    #print("B nchange is",scalinvar_distance(B,Bnew))
    B = Bnew
    ABdag = la.pinv(sla.khatri_rao(A,B)).T.reshape((Z.shape[0],Z.shape[1],A.shape[1]))
    Cnew = np.einsum("ijk,ijr->kr",Z,ABdag)
    #print("C change is",la.norm(C-Cnew))
    #print("C nchange is",scalinvar_distance(C,Cnew))
    if scalinvar_distance(C,Cnew) < 1e-07:
        flag =1
    C = Cnew
    return [A,B,C],flag

def ALS(guess,T,iters=20,fit_tol=0.99999,lamb=0):
    cp_rank = guess[0].shape[1]
    for i in range(iters):
        guess[0] = la.solve(((guess[1].T@guess[1])*(guess[2].T@guess[2]) +lamb*np.eye(cp_rank)).T,np.einsum('ijk,jr,kr->ir',T,guess[1],guess[2],optimize=True).T).T
        guess[1] =  la.solve(((guess[0].T@guess[0])*(guess[2].T@guess[2]) +lamb*np.eye(cp_rank)).T,np.einsum('ijk,ir,kr->jr',T,guess[0],guess[2],optimize=True).T).T
        guess[2] =  la.solve(((guess[0].T@guess[0])*(guess[1].T@guess[1]) +lamb*np.eye(cp_rank)).T,np.einsum('ijk,ir,jr->kr',T,guess[0],guess[1],optimize=True).T).T
        recon = np.einsum('ir,jr,kr->ijk',guess[0],guess[1],guess[2],optimize=True)
        res = la.norm (T -recon)
        if 1-(res/la.norm(T))>fit_tol:
            print('res is',res)
            print('converged after',i+1)
            break
        
    return guess

def colnormalize(A):
    B = np.zeros(A.shape)
    for i in range(A.shape[1]):
        B[:,i] = A[:,i]/la.norm(A[:,i])
        if B[0,i] < 0:
            B[:,i] *= -1
    return B

#def scalinvar_distance(A,B):
#    nsum = 0
#    for i in range(A.shape[1]):
#        nsum += (1-np.inner(A[:,i],B[:,i])/(la.norm(A[:,i])*la.norm(B[:,i])))/A.shape[1]
#    return nsum

def scalinvar_distance(A,B):
    nsum = 0
    for i in range(A.shape[1]):
        err1 = la.norm(A[:,i]/la.norm(A[:,i])+B[:,i]/la.norm(B[:,i]))
        err2 = la.norm(A[:,i]/la.norm(A[:,i])-B[:,i]/la.norm(B[:,i]))
        nsum += np.fmin(err1,err2)/A.shape[1]
    return nsum


def calibrate_norms(A,B,C):
    for i in range(A.shape[1]):
        anorm = la.norm(A[:,i])
        bnorm = la.norm(B[:,i])
        cnorm = la.norm(C[:,i])
        A[:,i] *= (bnorm*cnorm/anorm)**.5
        B[:,i] *= (anorm*cnorm/bnorm)**.5
        C[:,i] *= (anorm*bnorm/cnorm)**.5

def test_torth(T,U,V,W):
    r = U.shape[1]
    errs = []
    err = np.eye(r,r) - np.einsum("ijk,ir,js,ks->rs",T,U,V,W)
    errs.append(la.norm(err))
    err = np.eye(r,r) - np.einsum("ijk,is,jr,ks->rs",T,U,V,W)
    errs.append(la.norm(err))
    err = np.eye(r,r) - np.einsum("ijk,is,js,kr->rs",T,U,V,W)
    errs.append(la.norm(err))
    return np.sum(np.asarray(errs)**2)**.5

def test_stationary(T,U,V,W):
    r = U.shape[1]
    errs = []
    err = la.pinv(U) - np.einsum("ijk,js,ks->si",T,V,W)
    errs.append(la.norm(err))
    err = la.pinv(V) - np.einsum("ijk,is,ks->sj",T,U,W)
    errs.append(la.norm(err))
    err = la.pinv(W) - np.einsum("ijk,is,js->sk",T,U,V)
    errs.append(la.norm(err))
    return np.sum(np.asarray(errs)**2)**.5

def test_random_approx(T,U,V,W):
    r = U.shape[1]
    rU = U@np.random.random((U.shape[1]))
    rV = V@np.random.random((V.shape[1]))
    rW = W@np.random.random((W.shape[1]))
    errU = la.norm(np.einsum("ijk,j,k->i",T,rV,rW) - la.pinv(U @ ((V.T @ rV) * (W.T @ rW))))
    return errU


