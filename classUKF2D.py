import numpy as np 
       
def f(x, v, dt): # xk = f(xk) + vk 
    A = np.array([[1,0,dt,0], # transition Matrix same as in kf 
                  [0,1,0,dt],
                  [0,0,1,0],
                  [0,0,0,1]], float)
    
    G = np.array([[0.5*dt*dt, 0], #Process Noise Matrix same as in kf
                  [0, 0.5*dt*dt],
                  [dt, 0],
                  [0, dt]], float)
    return A @ np.asarray(x, float) + G @ np.asarray(v, float)

def h(x, n): # measure from sensors
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], float)
    return H @ np.asarray(x, float) + np.asarray(n, float)

class UKF2D:
    def __init__(self, dt, x0, P0, Pv, Pn, alpha=0.001, beta=2.0, kappa=0.0): 
     
       #constants
        self.dt = float(dt)
        self.dim_x = 4
        self.dim_v = 2
        self.dim_n = 2
        self.L =8  
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lamda = alpha**2 * (self.L + kappa) - self.L

        # short way to call to the fuction f and h
        # lambda is a short way to call the  outside function  
        self.fx = lambda x, v: f(x, v, self.dt)
        self.hx = lambda x, n: h(x, n)

        #the first values 
        self.x = np.asarray(x0, float)
        self.P = np.asarray(P0, float)
        self.process_noise = np.asarray(Pv, float)
        self.measurement_noise = np.asarray(Pn, float)

        # weights W
        self.c = self.L + self.lamda
        self.weights_mean = np.ones(2 * self.L + 1) * (1.0 / (2.0 * self.c))
        self.weights_cov = self.weights_mean.copy()
        self.weights_mean[0] = self.lamda / self.c
        self.weights_cov[0] = self.lamda / self.c + (1-alpha**2+beta)

    # xa = [x^T v^T n^T]^T
    def updatePxa(self):
        self.xa = np.concatenate([self.x,np.zeros(self.dim_v),np.zeros(self.dim_n)])
        # pa
        self.Pa = np.block([[self.P,np.zeros((self.dim_x, self.dim_v)), np.zeros((self.dim_x, self.dim_n))],
                           [np.zeros((self.dim_v, self.dim_x)), self.process_noise,np.zeros((self.dim_v, self.dim_n))],
                           [np.zeros((self.dim_n, self.dim_x)), np.zeros((self.dim_n, self.dim_v)), self.measurement_noise],])
        
   # covariance metrix must to be Symmetry         
    def symmetrize(self,M):
        return 0.5*(M + M.T)
    
    # Cholesky-Matrix Square Root
    def Cholesky1(self, M):
        return np.linalg.cholesky(M)

     # create sigma points (2L+1)
    def SigmaPoints(self, xa, Pa):
        S = self.Cholesky1(self.c * self.symmetrize(Pa))
        sigma_point = np.zeros((2*self.L + 1, self.L), dtype=float)
        sigma_point[0] = xa
        for i in range(self.L):
            d = S[:, i]
            sigma_point[i+1] = xa + d
            sigma_point[self.L+i+1] = xa - d
        return sigma_point
    
    
    # calculate predicted step
    def predict(self):
        self.updatePxa()
        Xa = self.SigmaPoints(self.xa, self.Pa)
        state_sigmas = Xa[:, :self.dim_x]
        process_noise_sigmas = Xa[:, self.dim_x:self.dim_x+self.dim_v]
        predicted_points_list = []
        for xk, vk in zip(state_sigmas, process_noise_sigmas):
         new_point = self.fx(xk, vk)
         predicted_points_list.append(new_point)

        pred_sigmas = np.array(predicted_points_list)
        x_pred = np.zeros(self.dim_x)
        for i in range(len(self.weights_mean)):
         w = self.weights_mean[i]
         point = pred_sigmas[i]
         x_pred += w * point

        P_pred = np.zeros_like(self.P)
        for i in range(pred_sigmas.shape[0]):
            dx = pred_sigmas[i] - x_pred
            P_pred += self.weights_cov[i] * np.outer(dx, dx)
        self.x = x_pred
        self.P = self.symmetrize(P_pred)
  
    # calculate update step  
    def update(self, z):
        z = np.asarray(z, float).reshape(self.dim_n,)
        self.updatePxa()
        Xa = self.SigmaPoints(self.xa, self.Pa)
        state_sigmas = Xa[:, :self.dim_x]
        Xn = Xa[:, self.dim_x + self.dim_v : self.dim_x + self.dim_v + self.dim_n]
        
        # Y_{k|k-1} = H[X^x_{k|k-1}, X^n_{k-1}]
        measurement_points_list = []
        for x_k, n_k in zip(state_sigmas, Xn):
            new_measurement = self.hx(x_k, n_k)
            measurement_points_list.append(new_measurement)
        Y = np.array(measurement_points_list)
        y_pred = np.zeros(self.dim_n)

        # y^_k^- = Sum( W_i^(m) * Y_{i, k|k-1} )
        for i in range(len(self.weights_mean)):
            weight = self.weights_mean[i]
            measurement_point = Y[i]
            y_pred += weight * measurement_point
        Syy = np.zeros((self.dim_n, self.dim_n)) 
        Pxy = np.zeros((self.dim_x, self.dim_n)) 
        
        # P_yy = Sum( W_i^(c) * [Y_i - y^_k^-][Y_i - y^_k^-]^T )
        # P_xy = Sum( W_i^(c) * [X_i - x^_k^-][Y_i - y^_k^-]^T )
        for i in range(Y.shape[0]):
            dy = Y[i] - y_pred
            dx = state_sigmas[i] - self.x  
            Syy += self.weights_cov[i] * np.outer(dy, dy)
            Pxy += self.weights_cov[i] * np.outer(dx, dy)

        # K = P_xy * (P_yy)^-1
        K = Pxy @ np.linalg.inv(self.symmetrize(Syy))

        check = z - y_pred 
        
        # x^_k = x^_k^- + K * (y_k - y^_k^-)
        self.x = self.x + K @ check
        
        # P_k = P_k^- - K * P_yy * K^T
        self.P = self.symmetrize(self.P - K @ Syy @ K.T)

