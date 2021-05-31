function ind = get_lowpass_inds_3d(k,siz)
if mod(siz(2),2)
    kxL = (siz(2)-1)/2;
else
    kxL = siz(2)/2;
end
if mod(siz(1),2) 
    kyL = (siz(1)-1)/2;
else
    kyL = siz(1)/2;
end
if mod(siz(3),2) 
    ktL = (siz(3)-1)/2;
else
    ktL = siz(3)/2;
end
    kxR = floor((siz(2)-1)/2);
    kyR = floor((siz(1)-1)/2);
    ktR = floor((siz(3)-1)/2);

ind = find((-kxL <= k(1,:)) & (k(1,:) <= kxR) & (-kyL <= k(2,:)) & (k(2,:) <= kyR)&(-ktL <= k(3,:)) & (k(3,:) <= ktR));
end
