close all;clear all;
%% Experiment Setup - data creation
% Simulation Meta Parametres
d = 10;             % # Features
L = 2;              % # Classes / Labels
num_repeats = 100;  % # samples data from wishart

mean_result = zeros(1,8);
for rr=1:8
    % Experiment Dependant Parametres
    alpha = 0.9;
    n_s = 1+25*rr;      % # source training data points
    n_t = 10;       % # target training data points
    k_t = 1;
    k_s = 1;
    k_ts = alpha*sqrt(k_t*k_s);

    % Prior Gaussian-Wishart Distribution
    M = zeros(d*2);
    M(1:d,1:d) = eye(d)*k_t;
    M(d+1:2*d,d+1:2*d)= eye(d)*k_s;
    M(1:d,d+1:2*d) = eye(d)*k_ts;
    M(d+1:2*d,1:d) = (eye(d)*k_ts)';
    m_t1 = zeros(d,1);
    m_t2 = ones(d,1)*0.05;
    m_s1 = m_t1 + ones(d,1);
    m_s2 = m_t2 + ones(d,1);
    kappa_t = 100;
    kappa_s = 100;
    nu=25;

    result = 0;
    for qq=1:num_repeats
        % Generate Synthetic Data
        PRES1 = wishrnd(M,nu);
        PRES1_t = PRES1(1:d,1:d);
        PRES1_s = PRES1(d+1:2*d,d+1:2*d);
        PRES1_ts = PRES1(1:d,d+1:2*d);

        PRES2 = wishrnd(M,nu);
        PRES2_t = PRES2(1:d,1:d);
        PRES2_s = PRES2(d+1:2*d,d+1:2*d);
        PRES2_ts = PRES2(1:d,d+1:2*d);

        mu_t1 = mvnrnd(m_t1,inv(kappa_t*PRES1_t),1);
        mu_t2 = mvnrnd(m_t2,inv(kappa_t*PRES2_t),1);
        mu_s1 = mvnrnd(m_s1,inv(kappa_s*PRES1_s),1);
        mu_s2 = mvnrnd(m_s2,inv(kappa_s*PRES2_s),1);

        X_t1 = mvnrnd(mu_t1,inv(PRES1_t),n_t);
        X_t2 = mvnrnd(mu_t2,inv(PRES2_t),n_t);
        X_s1 = mvnrnd(mu_s1,inv(PRES1_s),n_s);
        X_s2 = mvnrnd(mu_s2,inv(PRES2_s),n_s);

        % Test data generation (target only)
        y = 200;
        X_test = mvnrnd(mu_t1,inv(PRES1_t),y/2);
        X_test = [X_test;mvnrnd(mu_t2,inv(PRES2_t),y/2)];

        % Liklihood (not probability) of each label/class
        predictions = zeros(y,1);

        for j=1:y
            LH_1 = OBTL_labelwise(X_test(j,:),M,X_t1,X_s1,m_t1,m_s1,kappa_t,kappa_s);
            LH_2 = OBTL_labelwise(X_test(j,:),M,X_t2,X_s2,m_t2,m_s2,kappa_t,kappa_s);
            if LH_1 > LH_2
                predictions(j) = 1;
            else
                predictions(j) = 2;
            end
        end
        % Test to see how many predictions were correct
        error_rate = predictions ~= [repmat(1,y/2,1); repmat(2,y/2,1)];
        error_rate = sum(error_rate)/(y);
        result = result + error_rate;
    end
    result = result/num_repeats;
    mean_result(rr) = result;
end
mean_result
%% Target Functions
function class = OBTL_labelwise(x,M,X_t,X_s,m_t,m_s,kappa_t,kappa_s)
    % Priors
    nu = 25;
    
    d = size(M);
    d = d(1)/2;
    n_t = length(X_t);
    n_s = length(X_s);
    
    M_s = M(d+1:2*d,d+1:2*d);
    M_t = M(1:d,1:d);
    M_ts = M(1:d,d+1:2*d);
    C = M_s - M_ts'*inv(M_t)*M_ts;
    F = inv(C)*M_ts'*inv(M_t);
    

    S_t = out_prod_helper(X_t');
    S_s = out_prod_helper(X_s');
    T_t1 = inv(M_t) + F'*C*F + S_t + kappa_t*n_t/(kappa_t+n_t)*...
        (m_t-mean(X_t)')*(m_t-mean(X_t)')';
    T_s1 = inv(C) + S_s + kappa_s*n_s/(kappa_s+n_s)*...
        (m_s-mean(X_s)')*(m_s-mean(X_s)')';
    
    kappa_tn = kappa_t + n_t;
    kappa_sn = kappa_s + n_s;
    kappa_x = kappa_tn+1;
    m_tn = (kappa_t*m_t + sum(X_t)')/(kappa_t+n_t);
    m_sn = (kappa_s*m_s + sum(X_s)')/(kappa_s+n_s);
    
    T_x1 = T_t1 + kappa_tn/kappa_x * (m_tn-x')*(m_tn-x')';
    T_x = inv(T_x1);
    T_t = inv(T_t1);
    T_s = inv(T_s1);
    
%     class = pi^(-d/2)*(kappa_tn/kappa_x)^(d/2)*mvGamma(d,(nu+n_t+1)/2)*...
%         inv(mvGamma(d,(nu+n_t)/2))*det(T_x)^((nu+n_t+1)/2)*...
%         det(T_t)^(-(nu+n_t)/2)*...
%         gaussHyperGeometricApprox((nu+n_s)/2,(nu+n_t+1)/2,nu/2,T_s*F*T_x*F')*...
%         inv(gaussHyperGeometricApprox((nu+n_s)/2,(nu+n_t)/2,nu/2,T_s*F*T_t*F'));
%     class = pi^(-d/2)*(kappa_tn/kappa_x)^(d/2)*mvGamma(d,(nu+n_t+1)/2)*...
%         inv(mvGamma(d,(nu+n_t)/2))*det(T_x)^((nu+n_t+1)/2)*...
%         det(T_t)^(-(nu+n_t)/2)*...
%         mhg(55,2,[(nu+n_s)/2,(nu+n_t+1)/2],nu/2,diag(T_s*F*T_x*F')')*...
%         inv(mhg(55,2,[(nu+n_s)/2,(nu+n_t)/2],nu/2,diag(T_s*F*T_t*F')'));
    class = pi^(-d/2)*(kappa_tn/kappa_x)^(d/2)*mvGamma(d,(nu+n_t+1)/2)*...
        inv(mvGamma(d,(nu+n_t)/2))*det(T_x)^((nu+n_t+1)/2)*...
        det(T_t)^(-(nu+n_t)/2)*...
        Hypergeom2F1MatApprox((nu+n_s)/2,(nu+n_t+1)/2,nu/2,T_s*F*T_x*F')*...
        inv(Hypergeom2F1MatApprox((nu+n_s)/2,(nu+n_t)/2,nu/2,T_s*F*T_t*F'));
   
end
%% Helper Functions
function S = out_prod_helper(X)
    len = size(X);
    n = len(1);
    len = len(2);
    x_m = mean(X');
    
    t = zeros(n,len);
    for i=1:n
        t(i,:) = X(i,:)-x_m(i);
    end
    S = t*t';
end
function out = mvGamma(p,a)
    accum = 1;
    for i=1:p
        accum = accum * gamma(a+(1-i)/2);
    end
    out = accum*pi^(p*(p-1)/4);
end