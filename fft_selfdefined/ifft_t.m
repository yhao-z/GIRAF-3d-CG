function x=ifft_t(X)
%ʱ����ifft
x=ifft(size(X,3)*X,[],3);    
end
