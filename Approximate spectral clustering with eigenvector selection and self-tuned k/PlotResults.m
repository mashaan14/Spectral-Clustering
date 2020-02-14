%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML111
% Project Title: Neural Gas Network in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotResults(X, w, C)

    N = size(w,1);

    if ~isempty(X)
        plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
    end
    hold on;
    for i=1:N-1
        for j=i:N
            if C(i,j)>0
                plot([w(i,1) w(j,1)],[w(i,2) w(j,2)],'r','LineWidth',C(i,j));
            end
        end
    end
    plot(w(:,1),w(:,2),'ko','MarkerFaceColor','y','MarkerSize',7);
    hold off;
    axis off;
    pbaspect([1 1 1]); daspect([1 1 1]);
%     grid on;
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % gif utilities
%     set(gcf,'color','w'); % set figure background to white
%     drawnow;
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     outfile = 'outfile.gif';
%     % On the first loop, create the file. In subsequent loops, append.
%     if exist(outfile, 'file')==0
%         imwrite(imind,cm,outfile,'gif','DelayTime',0.3,'loopcount',inf);
%     else
%         imwrite(imind,cm,outfile,'gif','DelayTime',0.3,'writemode','append');
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end