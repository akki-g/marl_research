import numpy as np
import matplotlib.pyplot as plt

def a_id(action_vec):
    """
    Convert a binary action vector of shape (N,) into
    an integer index in [0, 2^N - 1].
    E.g., if action_vec = [0,1,0], that corresponds to index 2 (in base-2).
    """
    idx = 0
    for bit in action_vec:
        idx = idx * 2 + bit
    return idx

def id_to_vec(idx, N):
    """
    Inverse of a_id.
    Convert an integer idx in [0, 2^N - 1] into a binary vector of length N.
    """
    a = np.zeros(N, dtype=int)
    for i in reversed(range(N)):
        a[i] = idx % 2
        idx //= 2
    return a

def td_local_step(phi, pi_, w, beta, K, L, s, A, P, R_mean, gamma, w_star):
    """
    Python translation of the MATLAB function:
       [mse_it, w, s] = td_local_step(phi, pi, w, beta, K, L, s, A, P, R_mean, gamma, w_star)

    Inputs:
    -------
    phi     : shape (card_S, d) array of feature vectors
    pi_     : shape (N, card_S, card_Ai) array for policy
    w       : shape (d, N) initial weight parameters for each of N agents
    beta    : scalar step-size
    K       : integer number of local TD steps before each communication
    L       : integer number of communication rounds
    s       : integer current state index (0-based in Python)
    A       : shape (N, N) communication/consensus matrix
    P       : shape (card_S, card_S) Markov transition matrix
    R_mean  : shape (N, card_S, card_A) reward structure
    gamma   : discount factor
    w_star  : shape (d,) approximate solution from ODE

    Returns:
    --------
    mse_it  : 1D array of length L*K, MSE at each local step
    w       : shape (d, N) updated weight parameters after L communications
    s       : integer final state index
    """
    # Number of agents and dimension of each weight vector
    N = w.shape[1]
    d = w.shape[0]

    # Storage for actions, rewards, and TD errors
    a_storage = np.zeros((L, K, N), dtype=int)  # agent actions
    r_storage = np.zeros((L, K, N))            # rewards
    delta     = np.zeros((L, K, N))            # TD errors

    card_S = P.shape[0]
    mse_it = np.zeros(L * K)

    # Main loop over L communication rounds
    for l in range(L):
        # Increment array used in the local TD updates
        inc = np.zeros((d, N))

        # =====================
        # Local TD update loop
        # =====================
        for k in range(K):
            # Track MSE wrt w_star
            # (replicating norm(w - w_star*ones(1,N), 'fro') in MATLAB)
            diff = w - np.outer(w_star, np.ones(N))
            mse_it[l*K + k] = np.linalg.norm(diff, ord='fro')

            # Sample next state s_next (per the transition distribution P[s,:])
            seed = np.random.rand()
            cumulative = np.cumsum(P[s, :])
            s_next = np.searchsorted(cumulative, seed)

            # Sample actions for each agent from policy pi_
            for i in range(N):
                seed_a = np.random.rand()
                # In MATLAB, actions were [1,2]. Here we store them as [0,1]
                # but you can shift to [1,2] if you prefer.
                if seed_a < pi_[i, s, 0]:
                    a_storage[l, k, i] = 0  # "action = 1" in MATLAB
                else:
                    a_storage[l, k, i] = 1  # "action = 2" in MATLAB

            # Compute rewards and TD updates for each agent
            for i in range(N):
                # Convert the vector of actions for all agents
                # to an integer index for R_mean (0-based)
                ja_idx = a_id(a_storage[l, k, :])  # joint action index

                # r(l,k,i) = R_mean(i, s, ja_idx) + rand() - 0.5
                r_storage[l, k, i] = R_mean[i, s, ja_idx] + np.random.rand() - 0.5

                # delta(l,k,i) = r(l,k,i) + gamma * phi(s_next,:) * w(:,i)
                #               - phi(s,:) * w(:,i)
                delta[l, k, i] = (
                    r_storage[l, k, i]
                    + gamma * np.dot(phi[s_next, :], w[:, i])
                    - np.dot(phi[s, :], w[:, i])
                )

                # inc(:,i) += delta(l,k,i)*phi(s,:)
                inc[:, i] += delta[l, k, i] * phi[s, :]

                # w(:,i) = w(:,i) + beta*inc(:,i)
                w[:, i] += beta * inc[:, i]

            # Move to the next state
            s = s_next

        # ========================
        # Communication/Consensus
        # ========================
        temp_w = w.copy()
        w = np.zeros((d, N))
        for i in range(N):
            for j in range(N):
                w[:, i] += A[i, j] * temp_w[:, j]

    return mse_it, w, s


if __name__ == "__main__":

    # ================
    # Hyperparameters
    # ================
    N        = 20        # number of agents
    card_Ai  = 2         # cardinality of each agent's action space
    card_S   = 10        # cardinality of the global state space
    card_A   = card_Ai**N  # total joint actions
    rounds   = 10
    gamma    = 0.99
    d        = 5         # dimension of feature vector
    beta_1   = 0.005     # step size

    # Local step K and communication round L (paired as in MATLAB)
    K_list = [40,  50, 100, 200, 250]
    L_list = [250, 200, 100,  50,  40]

    sample_size = 10_000  # total sample size (L*K = 10^4)

    # =======================
    # Generate feature array
    # =======================
    # phi is shape (card_S, d)
    phi = np.random.rand(card_S, d)
    # Normalize each row
    for s in range(card_S):
        norm_s = np.linalg.norm(phi[s, :], 2)
        if norm_s > 0:
            phi[s, :] /= norm_s

    # ==========================
    # Generate transition matrix
    # ==========================
    P = np.random.rand(card_S, card_S)
    # Normalize each row
    for i in range(card_S):
        row_sum = np.sum(P[i, :])
        P[i, :] /= row_sum

    # ================================
    # Setup the network adjacency A
    # ================================
    if N > 1:
        diag_element = 0.4
        off_diag     = 0.3
        A = diag_element * np.eye(N)
        A[0, N - 1] = off_diag
        A[0, 1]     = off_diag
        for i in range(1, N - 1):
            A[i, i - 1] = off_diag
            A[i, i + 1] = off_diag
        A[N - 1, 0] = off_diag
        A[N - 1, N - 2] = off_diag
    else:
        A = np.array([[1.0]])

    # ====================
    # Setup the reward R
    # ====================
    # R_mean has shape (N, card_S, card_A)
    R_mean = np.zeros((N, card_S, card_A))
    for i in range(N):
        for s in range(card_S):
            for a_idx in range(card_A):
                R_mean[i, s, a_idx] = 4.0 * np.random.rand()

    # ================
    # Setup the policy
    # ================
    # pi has shape (N, card_S, card_Ai)
    pi = np.zeros((N, card_S, card_Ai))
    pi[:, :, 0] = 0.5
    pi[:, :, 1] = 0.5

    # ===================================================
    # Compute the stationary distribution 'dist' of P^T
    # ===================================================
    # Solve P' * x = x  =>  left eigenvector with eigenvalue 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Find eigenvalue closest to 1
    ix_1 = np.argmin(np.abs(eigvals - 1.0))
    dist_vec = np.real(eigvecs[:, ix_1])
    dist_vec = dist_vec / np.sum(dist_vec)  # normalized stationary distribution

    # diag_dist is a diagonal matrix of the stationary distribution
    diag_dist = np.diag(dist_vec)

    # ==========================
    # Compute r_s (value of r)
    # ==========================
    # r_s is shape (card_S,).  In MATLAB, r_s(s) accumulates average reward
    r_s = np.zeros(card_S)
    for s in range(card_S):
        # Probability weight for each joint action
        # from the product of pi(n, s, a_n)
        # plus the sum over all agents' immediate reward
        prob_a = np.ones(card_A)
        r_bar  = np.zeros(card_A)

        for a_idx in range(card_A):
            # decode joint action a_idx => array of length N
            act_vec = id_to_vec(a_idx, N)
            for n in range(N):
                prob_a[a_idx] *= pi[n, s, act_vec[n]]
                r_bar[a_idx]  += (1.0 / N) * R_mean[n, s, a_idx]
            r_s[s] += prob_a[a_idx] * r_bar[a_idx]

    # =============================================
    # Compute w_star by solving the "linear ODE"
    # =============================================
    # A_ode = phi' * dist * (phi - gamma P phi)
    # b_ode = phi' * dist * r_s
    # but phi', dist are arranged slightly differently in Python
    # so we do it carefully:
    # dist is diagonal => phi' * dist = phi.T @ diag_dist
    phi_T_dist = phi.T @ diag_dist
    # Then (phi - gamma*P*phi) => we need to do P*phi row-by-row
    P_phi = P @ phi
    A_ode = phi_T_dist @ (phi - gamma * P_phi)
    b_ode = phi_T_dist @ r_s
    # Solve A_ode * w_star = b_ode
    w_star = np.linalg.solve(A_ode, b_ode)

    # ======================
    # Prepare for the trials
    # ======================
    mse_it_1 = np.zeros((sample_size, rounds))
    mse_it_2 = np.zeros((sample_size, rounds))
    mse_it_3 = np.zeros((sample_size, rounds))
    mse_it_4 = np.zeros((sample_size, rounds))
    mse_it_5 = np.zeros((sample_size, rounds))

    for rnd in range(rounds):
        # Initialize the weight parameters and the starting state
        w_init = np.zeros((d, N))
        s_0 = np.random.randint(0, card_S)  # random state in [0, card_S-1]

        # Run each of the five K,L pairs
        mse1, w1, s_fin = td_local_step(phi, pi, w_init.copy(), beta_1,
                                        K_list[0], L_list[0],
                                        s_0, A, P, R_mean, gamma, w_star)
        mse_it_1[:, rnd] = mse1

        mse2, w2, s_fin = td_local_step(phi, pi, w_init.copy(), beta_1,
                                        K_list[1], L_list[1],
                                        s_0, A, P, R_mean, gamma, w_star)
        mse_it_2[:, rnd] = mse2

        mse3, w3, s_fin = td_local_step(phi, pi, w_init.copy(), beta_1,
                                        K_list[2], L_list[2],
                                        s_0, A, P, R_mean, gamma, w_star)
        mse_it_3[:, rnd] = mse3

        mse4, w4, s_fin = td_local_step(phi, pi, w_init.copy(), beta_1,
                                        K_list[3], L_list[3],
                                        s_0, A, P, R_mean, gamma, w_star)
        mse_it_4[:, rnd] = mse4

        mse5, w5, s_fin = td_local_step(phi, pi, w_init.copy(), beta_1,
                                        K_list[4], L_list[4],
                                        s_0, A, P, R_mean, gamma, w_star)
        mse_it_5[:, rnd] = mse5

    # =========================
    # Plot the averaged curves
    # =========================
    # Each MSE array is shape (sample_size, rounds). We take mean across columns.
    mse1_mean = np.mean(mse_it_1, axis=1) / (N * card_S)
    mse2_mean = np.mean(mse_it_2, axis=1) / (N * card_S)
    mse3_mean = np.mean(mse_it_3, axis=1) / (N * card_S)
    mse4_mean = np.mean(mse_it_4, axis=1) / (N * card_S)
    mse5_mean = np.mean(mse_it_5, axis=1) / (N * card_S)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, sample_size + 1), mse1_mean, 'r', label='K=40',  linewidth=2)
    plt.plot(range(1, sample_size + 1), mse2_mean, 'g', label='K=50',  linewidth=2)
    plt.plot(range(1, sample_size + 1), mse3_mean, 'b', label='K=100', linewidth=2)
    plt.plot(range(1, sample_size + 1), mse4_mean, 'm', label='K=200', linewidth=2)
    plt.plot(range(1, sample_size + 1), mse5_mean, 'c', label='K=250', linewidth=2)

    plt.legend()
    plt.xlabel('Sample Number')
    plt.ylabel('Objective Error')
    plt.title('TD Learning with Local Steps (Python Translation)')
    plt.grid(True)
    plt.show()
