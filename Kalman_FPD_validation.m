close all;clear all;
% Kalman Filter Transfer Learning Validation
% Conor Foley and Anthony Quinn. Fully probabilistic design for knowledge 
% transferin a pair of kalman filters.IEEE Signal Processing Letters, 
% 25(4):487–490, 2017
%% State Variables
n=70;

A = [1 1; 0 1];
H = [1 0];
Sig_x = 10^-5*eye(2);
Sig_z = 10^-3;
Sig_zs = 10^-5;

TNSE_transfer_KF = zeros(7,1);
TNSE_transfer_var_KF = zeros(7,1);
TNSE_single_KF = zeros(7,1);

%% Generate Data
x_n = zeros(2,n);
z_n = zeros(1,n);
z_ns = zeros(1,n);

x_n(:,1) = [2;3];
z_n(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_z);
z_ns(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_zs);

for i=2:n
    x_n(:,i) = A*x_n(:,i-1) + randn(2,1)*sqrt(10^-5);
    
    z_n(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_z);
    z_ns(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_zs);
end

%% Kalman filtering with transfer

mean_transfer = zeros(7,1);
for k=1:2000
    for j=1:7
        Sig_zs = 10^(-j);

        x_n = zeros(2,n);
        z_n = zeros(1,n);
        z_ns = zeros(1,n);

        x_n(:,1) = [2;3];
        z_n(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_z);
        z_ns(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_zs);

        for i=2:n
            x_n(:,i) = A*x_n(:,i-1) + randn(2,1)*sqrt(10^-5);

            z_n(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_z);
            z_ns(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_zs);
        end

        TNSE_transfer_KF(j) = 0;
        % Source Filter
        % Define one step ahead data predictor mean nu
        nu_is = zeros(1,n+1);

        % Data Step
        K_is = eye(2)*H'*(H*eye(2)*H' + Sig_zs)^-1;
        mu_is = ones(2,1)+K_is*(z_ns(1)-H*ones(2,1));
        D_is = (eye(2) - K_is*H)*eye(2);
        % Time Step
        m_is = A*mu_is;
        T_is = A*D_is*A' + Sig_x;
        % One step ahead data predictor
        nu_is(2) = H*m_is;
        nu_is(1) = H*m_is;

        for i=2:n
            % Data Step
            K_is = T_is*H'*(H*T_is*H' + Sig_zs)^-1;
            mu_is = m_is+K_is*(z_ns(i)-H*m_is);
            D_is = (eye(2) - K_is*H)*T_is;
            % Time Step
            m_is = A*mu_is;
            T_is = A*D_is*A' + Sig_x;
            % One step ahead data predictor
            nu_is(i+1) = H*m_is;
        end

        % Target Filter
        % Must do two iterations before we can start using transfered knowledge
        % Data step (1)
        K_i = eye(2)*H'*(H*eye(2)*H' + Sig_z)^-1;
        mu_i = ones(2,1)+K_i*(z_n(1)-H*ones(2,1));
        D_i = (eye(2) - K_i*H)*eye(2);

        % Time Step (2)
        m_i = A*mu_i;
        T_i = A*D_i*A'+Sig_x;
        % Data Step (2)
        K_i = T_i*H'*(H*T_i*H' + Sig_z)^-1;
        mu_i = m_i+K_i*(z_n(2)-H*m_i);
        D_i = (eye(2) - K_i*H)*T_i;

        for i=3:n
            % Time Step
            m_i = A*mu_i;
            T_i = A*D_i*A'+Sig_x;
            % Constraint step
            chi_i = T_i*H'*(H*T_i*H' + Sig_z)^-1;
            alp_i = m_i + chi_i*(nu_is(i) - H*m_i);
            E_i = (eye(2) - chi_i*H)*T_i;
            % Data Step
            K_i = E_i*H'*(H*E_i*H' + Sig_z)^-1;
            mu_i = alp_i+K_i*(z_n(i)-H*alp_i);
            D_i = (eye(2) - K_i*H)*E_i;
            if i> 20
                TNSE_transfer_KF(j) = TNSE_transfer_KF(j) + norm(mu_i-x_n(:,i))^2;
            end
        end
        mean_transfer(j) = mean_transfer(j) + TNSE_transfer_KF(j);
    end
end

%% Kalman filtering with transfer variant

mean_transfer_var = zeros(7,1);
for k=1:2000
    for j=1:7
        % Generate data
        Sig_zs = 10^(-j);

        x_n = zeros(2,n);
        z_n = zeros(1,n);
        z_ns = zeros(1,n);

        x_n(:,1) = [2;3];
        z_n(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_z);
        z_ns(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_zs);

        for i=2:n
            x_n(:,i) = A*x_n(:,i-1) + randn(2,1)*sqrt(10^-5);

            z_n(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_z);
            z_ns(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_zs);
        end

        TNSE_transfer_var_KF(j) = 0;
        % Source Filter
        % Define one step ahead data predictor mean nu
        nu_is = zeros(1,n+1);
        phi_is = zeros(1,n+1);

        % Data Step
        K_is = eye(2)*H'*(H*eye(2)*H' + Sig_zs)^-1;
        mu_is = ones(2,1)+K_is*(z_ns(1)-H*ones(2,1));
        D_is = (eye(2) - K_is*H)*eye(2);
        % Time Step
        m_is = A*mu_is;
        T_is = A*D_is*A' + Sig_x;
        % One step ahead data predictor
        nu_is(2) = H*m_is;
        nu_is(1) = H*m_is;
        phi_is(2) = H*T_is*H' + Sig_zs;
        phi_is(1) = H*T_is*H' + Sig_zs;

        for i=2:n
            % Data Step
            K_is = T_is*H'*(H*T_is*H' + Sig_zs)^-1;
            mu_is = m_is+K_is*(z_ns(i)-H*m_is);
            D_is = (eye(2) - K_is*H)*T_is;
            % Time Step
            m_is = A*mu_is;
            T_is = A*D_is*A' + Sig_x;
            % One step ahead data predictor
            nu_is(i+1) = H*m_is;
            phi_is(i+1) = H*T_is*H' + Sig_zs;
        end

        % Target Filter
        % Must do two iterations before we can start using transfered knowledge
        % Data step (1)
        K_i = eye(2)*H'*(H*eye(2)*H' + Sig_z)^-1;
        mu_i = ones(2,1)+K_i*(z_n(1)-H*ones(2,1));
        D_i = (eye(2) - K_i*H)*eye(2);

        % Time Step (2)
        m_i = A*mu_i;
        T_i = A*D_i*A'+Sig_x;
        % Data Step (2)
        K_i = T_i*H'*(H*T_i*H' + Sig_z)^-1;
        mu_i = m_i+K_i*(z_n(2)-H*m_i);
        D_i = (eye(2) - K_i*H)*T_i;


        for i=3:n
            % Time Step
            m_i = A*mu_i;
            T_i = A*D_i*A'+Sig_x;
            % Constraint step
            chi_i = T_i*H'*(H*T_i*H' + phi_is(i))^-1;
            alp_i = m_i + chi_i*(nu_is(i) - H*m_i);
            E_i = (eye(2) - chi_i*H)*T_i;
            % Data Step
            K_i = E_i*H'*(H*E_i*H' + Sig_z)^-1;
            mu_i = alp_i+K_i*(z_n(i)-H*alp_i);
            D_i = (eye(2) - K_i*H)*E_i;
            if i>20
                TNSE_transfer_var_KF(j) = TNSE_transfer_var_KF(j) + norm(mu_i-x_n(:,i))^2;
            end
        end
        mean_transfer_var(j) = mean_transfer_var(j) + TNSE_transfer_var_KF(j);
    end
end
%% Singular Kalman filter just target
mean_single = zeros(7,1);
for k=1:2000
    for j=1:7
        Sig_zs = 10^(-j);
        x_n = zeros(2,n);
        z_n = zeros(1,n);
        z_ns = zeros(1,n);

        x_n(:,1) = [2;3];
        z_n(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_z);
        z_ns(1) = H*x_n(:,1) + randn(1,1)*sqrt(Sig_zs);

        for i=2:n
            x_n(:,i) = A*x_n(:,i-1) + randn(2,1)*sqrt(10^-5);

            z_n(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_z);
            z_ns(i) = H*x_n(:,i) + randn(1,1)*sqrt(Sig_zs);
        end

        TNSE_single_KF(j) = 0;

        % Data Step
        K_i = eye(2)*H'*(H*eye(2)*H' + Sig_z)^-1;
        mu_i = ones(2,1)+K_i*(z_n(1)-H*ones(2,1));
        D_i = (eye(2) - K_i*H)*eye(2);
        % Time Step
        m_i = A*mu_i;
        T_i = A*D_i*A' + Sig_x;

        for i=2:n
            % Data Step
            K_i = T_i*H'*(H*T_i*H' + Sig_z)^-1;
            mu_i = m_i+K_i*(z_n(i)-H*m_i);
            D_i = (eye(2) - K_i*H)*T_i;
            % Time Step
            m_i = A*mu_i;
            T_i = A*D_i*A' + Sig_x;

            if i > 20
                TNSE_single_KF(j) = TNSE_single_KF(j) + norm(mu_i-x_n(:,i))^2;
            end
        end
        mean_single(j) = mean_single(j) + TNSE_single_KF(j);
    end
end
%% Graph MC results
mean_single_Log = log10(mean_single/2000);
mean_transfer_Log = log10(mean_transfer/2000);
mean_transfer_var_Log = log10(mean_transfer_var/2000);

scatter(-1:-1:-7,mean_single_Log)
hold on
scatter(-1:-1:-7,mean_transfer_Log)
hold on
scatter(-1:-1:-7,mean_transfer_var_Log)