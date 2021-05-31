clear;
clc;
%% load data
addpath('./data','./etc','./fft_selfdefined');
load acc=10_192_radialray.mat
load data25.mat
x_origin=data;
% x0=fftn(x_origin)/numel(x_origin);
x0=fft3(x_origin); %FFT频域
sampmask=mask0;
res=size(x0);
%% define index sets
ind_samples = find(sampmask);
[S,St] = defSSt(ind_samples,res); %采样算子，逆算子（未采样点填0）
b = S(ifft_t(x0)); %时间轴ifft，之后在二维层面欠采样
xinit = fft_t(St(b)); %xinit欠采样后，完全的频域，二维以及时间轴
%% global problem settings
settings.filter_siz = [5 5 3];
settings.res = size(x0);
settings.exit_tol = 1e-4;  %exit algorithm if relative change in NRMSE between iterates less than this
settings.lambda = 1; %regularization parameter 
settings.p = 0; %Schatten p penalty (0 <= p <= 1)
%% GIRAF parameters
param.iter = 30; %number of IRLS iterations
param.eps0 = 0; %inital epsilon (0=auto-init) 
param.eta = 1.5; %epsilon decrease factor (typically between 1.1-1.5);
param.epsmin = 1e-7;
param.ADMM_iter = 30;
param.ADMM_tol = 1e-4;
param.delta = 1; %ADMM conditioning parameter (typically between 10-100);
param.overres = settings.res + 2*settings.filter_siz;
%% run GIRAF
[x,cost] = giraf_3d(x_origin,xinit,b,S,sampmask,param,settings);
SNR = -20*log10(norm(x(:)-x0(:))/norm(x0(:)));
