function dz = get_kspace_weights_3d(k,res)
%% һ�׵������Ӷ���
dz = zeros([res,3]);
dz(:,:,:,1) = reshape(1j*2*pi*k(1,:),res)/res(2);
dz(:,:,:,2) = reshape(1j*2*pi*k(2,:),res)/res(1);
% dz(:,:,:,3) = reshape(1j*2*pi*k(3,:),res)/res(3);
dz(:,:,:,3) = reshape(1j*2*pi*k(3,:),res); %����1
end
