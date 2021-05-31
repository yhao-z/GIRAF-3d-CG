function X=fft3(x)
%% »˝Œ¨°¢ÀƒŒ¨æÿ’ÛFFT
siz=size(x);
X=zeros(siz);
if(length(siz)==3)
    X=fftn(x)/prod(siz);
elseif(length(siz)==4)
    for i=1:siz(4)
        X(:,:,:,i)=fftn(x(:,:,:,i))/prod(siz(1:3));
    end
else
    error(['the fft3 function cannot apply to ',num2str(length(siz)),' dimentional input data']);
end
end
