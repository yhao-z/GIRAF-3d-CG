function [x,cost] = giraf_3d(x0,xinit,b,S,sampmask,param,settings)
%GIRAF run GIRAF 3D algorithm
%   Input:
%       x0: ground truth of 3d MRI data ��ֵ
%       xinit: initialization of x (in Fourier domain) 
%       b: vector of sampled entries
%       S: function handle for sampling operator
%       St: function handle for transpose sampling operator
%       sampmask: logical mask of sampling locations (in respect to k-space)
%       param: struct of GIRAF-specific parameters 
%           (see specification of fields in code below)
%       settings: struct of gloabl problem settings
%           (see specification of fields in code below)
%       
%   Output:
%       x: reconstructed Fourier data
%       cost: values of cost function per iteration
%
%   GIRAF solves the optimization problem:
%   min_x ||Ax-b||_2^2 + lambda*||T(x)||_p^p
%   where A is a function that consists of inverse Fourier transform at t (time) dimention Ft 
%   and a sampling operator S. In formula, A=Ft*S
%   T(x) is a Toeplitz-like matrix built from 3d fourier coefficient data x
%   and ||.||_p is the Schatten p quasi-norm, with 0 <= p <= 1.
%
%   For more details see the 2d GIRAF paper, and extend it to 3d:
%   A Fast Algorithm for Convolutional Structured Low-Rank Matrix Recovery
%   G. Ongie & M. Jacob, 2017. 
%   Pre-print available online: 
%   https://arxiv.org/abs/1609.07429
%
%   Note:
%   this work is an extention from Greg Ongie 2d GIRAF code to 3d GIRAF.
%   2d GIRAF by Greg Ongie can be find on github: 
%   https://github.com/cbig-iowa/giraf
%
%  Yinghao Zhang 31/10/2020
%
%  ��������ע�ͣ�ʵ������

p = settings.p; %Schatten p value
q = 1-(p/2);    
lambda = settings.lambda; %regularization parameter
filter_siz = settings.filter_siz; %rectangular filter dimensions
res = settings.res; %recon pixel resolution
filter_siz2 = 2*filter_siz - [1,1,1]; %squared filter dimensions

iter = param.iter; %number of iterations
eta = param.eta;   %epsilon decrease factor;
eps = param.eps0;  %initial epsilon
epsmin = param.epsmin; %minimum epsilon value
ADMM_iter = param.ADMM_iter; %number of inner ADMM iterations
ADMM_tol = param.ADMM_tol;   %exit tolerance for ADMM
if(isfield(param,'delta'))   %ADMM conditioning parameter
    delta = param.delta;
else
    delta = 100; %default %in GIRAF paper, it was 10
end

if(isfield(param,'eps0')) %intial epsilson
    eps0 = param.eps0;
else
    eps0 = 0; %auto-initialize option
end

if(isfield(param,'overres')) %oversampled reconstruction grid
    overres = param.overres;
else
    overres = res + 2*filter_siz; %default
end

%intialize variables, operators, and index sets
k = get_kspace_inds_3d(overres);
ind_full = get_lowpass_inds_3d(k,res);
ind_filter = get_lowpass_inds_3d(k,filter_siz);
ind_filter2 = get_lowpass_inds_3d(k,filter_siz2);

dz = get_kspace_weights_3d(k,overres);

M = @(z) repmat(z,[1,1,1,size(dz,4)]).*dz; % MΪ�ڵ���ά��չ�Ľ�������ۺ��˶�x,y,t����ƫ���Ŀ��ټ�����󣬷�ֹforѭ�����ӿ��ٶȡ�
Mt = @(Z) sum(Z.*conj(dz),4);  %M_tranpose
MtM = Mt(M(ones(overres))); %������ԽǾ�������ԣ���˿��������гɾ������㣬������Ӽ�����

x = xinit;
x_pad = zeros(overres);
x_pad(ind_full) = x; %��ȫ��Ƶ��󣬲���

FtStb_pad=x_pad;%��ȫ��Ƶ��󣬲���

cost = zeros(1,iter);
fprintf('Starting GIRAF (p=%d, lambda = %1.1e)\n',p,lambda);
tic;
for i=1:iter 
    %step 1: Compute sos annihilating polynomial
    gradx = M(x_pad);
    G = build_gram_matrix(gradx,filter_siz,filter_siz2,ind_filter2);
    [U,S_ev] = eig(G);
    ev = abs(diag(S_ev));
    if i==1 %initialze epsilon
        if eps0 > 0 
            eps = eps0;
        else
            eps = 0.001*max(ev); %auto-init eps %�����н����ǳ���100���˴���1000
        end
    end
    mu = build_sos_poly(U,ev+eps,q,overres,filter_siz,filter_siz2,ind_filter,ind_filter2); %d�ļ���

    %step 2: ADMM solution of least squares problem
    gam = max(mu(:))/delta;  %set ADMM parameter some fraction of sos-mask max value
    [x_pad,~] = run_ADMM_WL2(x_pad,res,overres,ind_full,mu,M,Mt,sampmask,MtM,FtStb_pad,gam,lambda,ADMM_iter,ADMM_tol);
    x_pad = reshape(x_pad,overres);
    x_prev = x;
    x = reshape(x_pad(ind_full),res); %ȥ�����
    Xr = ifft3(x); %IFFT��ʱ��

    %update epsilon
    eps = max(eps/eta,epsmin);

    %cost computations (of previous iterate)
    if p == 0
        shatten = 0.5*sum(log(ev+epsmin));
    else
        shatten = (1/p)*sum((ev+epsmin).^(p/2));
    end        
    diff = S(ifft_t(x_prev))-b;
    cost(i) = norm(diff(:)).^2 + lambda*shatten; 
    figure(10);plot(cost);drawnow;

    %check stopping condition
    if i>1
        if abs(cost(i)-cost(i-1))/abs(cost(i))<=1e-9
            fprintf('**Reached exit tolerance: relerr < %2.2e\n',settings.exit_tol);
            break;
        end
    end
    
    SNR_iter = 20*log10(norm(Xr(:))/norm(Xr(:)-x0(:)));
    fprintf('iter: %d----> SNR: %6.4f \n',i,SNR_iter);
%     fprintf('Finished iteration %d of %d (cost=%2.2e)\n',i,iter,cost(i));
end
runtime = toc;
fprintf('Done! Total runtime: %6.1f s\n',runtime);
fprintf('Final cost: %2.3e\n',cost(i));
fprintf('\n');

end

%ADMM weighted least squares solver
%solves ||Ax-b||_2^2 + lambda*||D^{1/2}Mx||_2^2
%where D = diag(mu)
function [x,resvec] = run_ADMM_WL2(x0,res,overres,ind_full,mu,M,Mt,sampmask,MtM,FtStb_pad,gam,lambda,iter,tol)
    x = x0;
    Mx = M(x);
    L = zeros(size(Mx));
    ndz = size(L,4);
    resvec = zeros(1,iter);
    for ii = 1:iter
        % Y subprob 
        Z = gam*(Mx+L);
        muinv = repmat((mu + gam).^(-1),[1,1,1,ndz]);%�˴�������gam����Ҫ��mu����ͬһ�����£�̫������ʹ�����ɵ�mu��y�Ĺ��׼�������
        Y = fft3(muinv.*ifft3(Z));   %ע��Y�Ķ����������е�y����ͬ��Y=fft3(y)

        % x subprob
        A_CG = @(z) A_CG_Func(z,res,overres,ind_full,sampmask,MtM,lambda,gam); %����CG�е�A���ӣ������ǹ�ʽ�е�A=Ft*S
        b_CG = FtStb_pad+lambda*gam*Mt(Y-L);%����CG�е�b
        x = pcg(A_CG,b_CG(:),1e-12,50,[],[],x(:));
        x = reshape(x,overres);

        % L update%��Ϊ�����е�Q
        Mx = M(x);
        residue = Mx-Y;
        L = L + residue;

        resvec(ii) = norm(residue(:))/norm(Y(:));
        if (iter > 10) && (resvec(ii) < tol)
            return;
        end
    end
    
    function z_vector=A_CG_Func(z_pad,res,overres,ind_full,sampmask,MtM,lambda,gam)
    %ע�������pad��Ӱ�죬����ڼ���AʱҪ����pad�������Ƶ���ʽʱ��pad��Ϊһ������
        z_pad=reshape(z_pad,overres);
        z=reshape(z_pad(ind_full),res);
        z=fft_t(sampmask.*ifft_t(z)); %ifftǷ����Ȼ����fft
        z1=zeros(overres);
        z1(ind_full)=z;
        
        z2=lambda*gam*MtM.*(z_pad);
        
        z_vector=z1(:)+z2(:);
    end

end


%Function to build sum-of-squares annihilation weights
function mu = build_sos_poly(U,s,q,overres,filter_siz,filter_siz2,ind_filter,ind_filter2)
%ʵ���ϴ˲����ǽ�overres��С�ĸ���Ҷ�任�ļ�������С��fiter_siz2��С��fft�������������ǽ�N(����ֵ����)�θ���Ҷ�任ת��Ϊһ�θ���Ҷ�任
%Ҫ���������е��Ƶ����ص�������⡰���������������FFT�������������������˵���Եġ�

%     normfac = prod(overres)/prod(filter_siz);
%     %ԭ����������Ľ�������Ǿ����Ƶ����ֲ�����fiter_siz��Ӧ����fiter_siz2
    normfac = prod(overres)/prod(filter_siz2);
    mu_small = zeros(filter_siz2);
    for j=1:length(s)
        filter_pad = zeros(overres);
        filter_pad(ind_filter) = ifftshift(reshape(U(:,j),filter_siz)); %����G������Forierϵ������˴˴�Ҫshiftһ��
        filter = reshape(filter_pad(ind_filter2),filter_siz2);
        mu_small = mu_small + ((1/s(j))^q)*(abs(ifft3(filter)).^2);
    end
    muhat_small = fft3(mu_small);
    muhat = zeros(overres);
    muhat(ind_filter2) = muhat_small;
    mu = ifft3(muhat)/normfac;
end


% Function to build gram matrix G=T(x)^*T(x) 
% using ffts and neighborhood operators
function G = build_gram_matrix(gradx,filter_siz,filter_siz2,ind_filter2)
    gradx_ifft = ifft3(gradx);
    sos = fft3(sum(abs(gradx_ifft).^2,4));
    sos2 = fftshift(reshape(sos(ind_filter2),filter_siz2));  %Ҫ�����任�������ĸ���Ҷ��Ȼ����о���л������򽫲�����ԭ����Ƶ�
    G = im2colstep(real(sos2),filter_siz)+1i.*im2colstep(imag(sos2),filter_siz);%ʹ�õĺ�����C��д�ģ�����etc�ļ����У�ֻ֧��ʵ�����㣬�����ɸĽ�
    G = rot90(G,-1);%����������Ҷ��Ľ�����м����������
end



