function k = get_kspace_inds_3d( res )
if mod(res(2),2)
    indx = ifftshift(-((res(2)-1)/2):((res(2)-1)/2));
else
    indx = [0:((res(2)/2)-1), -(res(2)/2):-1];
end
if mod(res(1),2) 
    indy = ifftshift(-((res(1)-1)/2):((res(1)-1)/2));
else
    indy = [0:((res(1)/2)-1), -(res(1)/2):-1];
end
if mod(res(3),2) 
    indt = ifftshift(-((res(3)-1)/2):((res(3)-1)/2));
else
    indt = [0:((res(3)/2)-1), -(res(3)/2):-1];
end
[kx,ky,kt] = meshgrid(indx,indy,indt);
k(1,:) = kx(:);
k(2,:) = ky(:);
k(3,:) = kt(:);
end
