function x=ifft3(X)
%% »˝Œ¨°¢ÀƒŒ¨æÿ’ÛIFFT
siz=size(X);
x=zeros(siz);
if(length(siz)==3)
    x=ifftn(prod(siz)*X);
elseif(length(siz)==4)
    for i=1:siz(4)
        x(:,:,:,i)=ifftn(prod(siz(1:3))*X(:,:,:,i));
    end
else
    error(['the fft3 function cannot apply to ',num2str(length(siz)),' dimentional input data']);
end
end
