import numpy as np
import casadi as ca
import simulation as sim
import time

class NMPC:
    def __init__(self, p, m, steps, nY, nX, nU, Q, R, dt, iter):
        # Constantes iniciais
        self.p = p # Horizonte de predição
        self.m = m # Horizonte de controle
        self.steps = steps # Passos no passado
        self.nU = nU # Número de entradas
        self.nX = nX # Número de estados
        self.nY = nY # Número de saídas
        self.Q = Q # Matriz de peso das saídas
        self.R = R # Matriz de peso das entradas
        self.iter = iter # Quantidade de iterações
        self.dt = dt # Tempo entre cada ponto simulado
        # Tempo total simulado = iter * dt

        self.dU = np.zeros((nU * m, 1)) 

        # Objetos para simulação
        self.sim_pred = sim.RiserModel(p, m, steps, nY, nX, nU, dt)
        self.sim_mf = sim.RiserModel(1, 1, steps, nY, nX, nU, dt)
        
        # Setpoints
        self.SPList = [[9.5e6, 9.74e6, 0, 0], [9.16e6, 9.5e6, 0, 0]]
        
        # Poço 1 min = 9.16e6 max = 9.52
        # Poço 2 min = 9.41e6 max = 9.74
        
        print(self.SPList)
        self.y_sp = np.array(self.SPList[0])
        
        # TODO: Adicionar restrições de entrada e estado
        self.u_min = np.array([[0.1], [0.1]])
        self.u_max = np.array([[5], [5]])
        self.dU_min = np.array([[-1], [-1]])
        self.dU_max = np.array([[1], [1]])
        self.y_min = np.array([[0] for _ in range(nY)])
        self.y_max = np.array([[np.inf] for _ in range(nY)])
    
    def iTil(self, n, x):
        n = np.tile(n,(x,1))
        return n
    
    def diagMatrix(self, x,n):
        x = np.float64(x)
        n = int(n)
        X_matrix = np.full((n,n),0, dtype=np.float64)
        np.fill_diagonal(X_matrix,x)
        return X_matrix
    
    def matriz_triangular_identidade(self, m, n, N):
        matriz = np.zeros((m * N, n * N))
        
        for i in range(m):
            for j in range(n):
                if j <= i:
                    matriz[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.eye(N)
        
        return ca.DM(matriz)  # Convertendo para CasADi DM
    
    def ajusteMatrizes(self):
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1)) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
        self.y_max = ca.DM(self.iTil(self.y_max,self.p)) # Expansão do y_max para P. SHAPE -> (nY*P, 1)
        self.y_min = ca.DM(self.iTil(self.y_min,self.p)) # Expansão do y_min para P. SHAPE -> (nY*P, 1)

        self.u_min = ca.DM(self.iTil(self.u_min,self.m)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.m)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

        self.dU_min = self.iTil(self.dU_min, self.m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
        #self.dU_min = ca.DM(np.concatenate((self.dU_min,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
        self.dU_max = self.iTil(self.dU_max, self.m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
        #self.dU_max = ca.DM(np.concatenate((self.dU_max,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

        # Supondo que self.Q tem shape (nY, 1)
        q_tile = np.array(self.Q * self.p)  # Repete p vezes → shape (nY*p,)
        self.Q = ca.DM(np.diag(q_tile))  # Cria matriz diagonal (nY*p, nY*p)

        r_tile = np.array(self.R * self.m)  # Repete m vezes → shape (nU*m,)
        self.R = ca.DM(np.diag(r_tile))  # Cria matriz diagonal (nU*m, nU*m)

    def nlp_func(self):
        # Criando o problema de otimização
        opti = ca.Opti()

        # Definição das variáveis de decisão
        dUs = opti.variable(self.nU * self.m, 1)
        Fs  = opti.variable(1, 1)  # Variável escalar para Fs
        yModelk = opti.parameter(self.nY * self.steps, 1)
        xModelk = opti.parameter(self.nX * self.steps, 1)  # xModelk como parâmetro
        uModelk = opti.parameter(self.nU * self.steps, 1)
        yPlantak = opti.parameter(self.nY, 1)
        ysp = opti.parameter(self.nY * self.p, 1)# yPlantak como parâmetro 
    
        x = ca.vertcat(dUs, Fs)
        
        # Definição do problema de otimização
        opti.minimize(Fs)

        # Erro entre a planta e o modelo
        dYk = yPlantak - yModelk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)

        # Predição do modelo
        yModel_pred, _, _ = self.sim_pred.caPredFun(xModelk, uModelk, dUs)

        # Matriz triangular para os controles
        matriz_inferior = self.matriz_triangular_identidade(self.m, self.m, self.nU)

        # Restrições
        # x_min e x_max
        opti.subject_to(opti.bounded(self.dU_min, dUs, self.dU_max))
        opti.subject_to(opti.bounded(0, Fs, 10e23))

        # lbg e ubg
        opti.subject_to(opti.bounded(self.y_min, yModel_pred + dYk, self.y_max))
        opti.subject_to(opti.bounded(self.u_min, ca.repmat(uModelk[-self.nU:], self.m, 1) + matriz_inferior @ dUs, self.u_max))
        opti.subject_to(Fs - ((yModel_pred - ysp + dYk).T @ self.Q @ (yModel_pred - ysp + dYk) + dUs.T @ self.R @ dUs) == 0)  # Restrições de igualdade

        opti.solver('ipopt', {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-6,
            "ipopt.linear_solver": "mumps"
        })
        print(opti)

        # Criando a função otimizada
        return opti.to_function(
            "opti_nlp",
            [yModelk, xModelk, uModelk, yPlantak, ysp, dUs, Fs],
            [x]
        )

    def otimizar(self, ymk, xmk, umk, ypk):
        # Valores iniciais antes da otimização
        dYk = ypk - ymk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)
        dU_init = self.dU
        yModel_init, _, _ = self.sim_pred.caPredFun(xmk, umk, dU_init)
        yModel_init = np.array(yModel_init.full())
        Fs_init = (yModel_init - self.y_sp + dYk).T @ self.Q @ (yModel_init - self.y_sp + dYk) + dU_init.T @ self.R @ dU_init

        # Otimização
        x_opt = self.opti_nlp(ymk, xmk, umk, ypk, self.y_sp, dU_init, Fs_init)

        dU_opt = x_opt[:self.nU * self.m]
        dU_opt = np.array(dU_opt.full())
        return dU_opt
    
    def run(self):
        # Ajuste matricial
        self.ajusteMatrizes()
        
        # Valores iniciais
        '''Aqui os valores são preparados
        os valores que têm 'm' são valores para o modelo, 'p' para a planta
        os valores com _next são os valores que serão aplicados nas listas para print'''
        ymk, xmk, umk = self.sim_pred.pIniciais()
        ypk = ymk[-self.nY:]
        ymk_next = ypk
        xpk = xmk[-self.nX:]
        xmk_next = xpk
        upk = umk[-self.nU:]
        umk_next = upk

        # Lei de controle
        self.opti_nlp = self.nlp_func()

        # Listas para plot
        Ypk_forPrint = []
        Upk_forPrint = []
        dU_forPrint = []
        Ymk_forPrint = []
        Ysp_forPrint = []
        Tempos_forPrint = []

        # Loop de controle
        for i in range(self.iter): 
            t1 = time.time()
            print(15*'='+ f'Iteração {i+1}' + 15*'=')
            # Otimização e predição dos passos de controle
            dU_opt = self.otimizar(ymk, xmk, umk, ypk)
            
            self.dUk = dU_opt[:self.nU]
            self.dU = dU_opt
            
            # Aplica o primeiro passo de controle nas entradas
            umk = umk.reshape(self.steps*self.nU, 1)
            umk = np.append(umk, umk[-self.nU:] + self.dUk)
            umk = umk[self.nU:]

            # Simulação de apenas 1 passo pelo modelo
            ymk_next, xmk_next, umk_next = self.sim_mf.caPredFun(xmk[-self.nX:], umk[-self.nU:], [0,0])
            ymk_next = np.array(ymk_next.full())
            xmk_next = np.array(xmk_next.full())
            umk_next = np.array(umk_next.full())
            
            t2 =  time.time()
            Tempos_forPrint.append(t2-t1)
            print(f'Tempo decorrido: {t2-t1}')
            
            # Simulação de apenas 1 passo pela planta
            ypk, xpk, upk = self.sim_mf.pPlanta(xpk, self.dUk)

            # Diferença entre a simulação da planta e modelo
            print('dYk: ',ymk_next - ypk)
            
            # Atualizações de variáveis para o próximo loop
            upk = upk.flatten()
            xpk = xpk.flatten()
            ypk = ypk.flatten()
            
            ymk = np.append(ymk, ymk_next)
            ymk = ymk[self.nY:]
            xmk = np.append(xmk, xmk_next)
            xmk = xmk[self.nX:]
            umk = np.append(umk, umk_next)
            umk = umk[self.nU:]

            # Variáveis para visualização
            Ymk_forPrint.append(ymk_next.copy())
            Ypk_forPrint.append(ypk.copy())
            Upk_forPrint.append(upk.copy())
            dU_forPrint.append(self.dUk.copy())
            print('dU_opt: ',dU_opt[:self.m*self.nU])
            print('dUk: ', self.dUk)
            print('Uk: ', upk)
            Ysp_forPrint.append(np.array(self.y_sp.full()).copy())
            
            # Mudança de setpoint
            set_index = i // (self.iter // len(self.SPList))
            set_index = min(set_index, len(self.SPList)-1)
            self.y_sp = np.array(self.SPList[set_index])
            self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
        
        return self.iter, Ymk_forPrint, Ypk_forPrint, Upk_forPrint, dU_forPrint, Ysp_forPrint, Tempos_forPrint, self.dt