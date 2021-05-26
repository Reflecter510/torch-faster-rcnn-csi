function segment=segdata_display(med_time,filtered_amplitude,i,fileName)%pca_phase2,
resultSavePath='D:\my documents\AI+IOT\dataset2019-11-15\S2-P1-O3-figure\seg\';
close all
slide_window=20; 

%-------------segment the series---------------------------  
  amplitude=med_time;% pca_phase=pca_phase2;
  amplitude=(amplitude-min(amplitude))/(max(amplitude)-min(amplitude));
%  pca_phase=(pca_phase-min(pca_phase))/(max(pca_phase)-min(pca_phase));
  diff_amp = diff(amplitude);%diff_phase=diff(pca_phase);
  subplot(2,1,1);plot(amplitude)
%   subplot(3,1,2);plot(pca_phase)
  subplot(2,1,2);plot(filtered_amplitude)
  %         figure;plot(diff_amp)
  begin=[];s=[];e=[];
  for slide=1:1:length(diff_amp)-slide_window
      tmp_amp=mean(abs(diff_amp(slide:slide+slide_window-1)));
%      tmp_phase=mean(abs(diff_phase(slide:slide+slide_window-1)));
      %             tmp1=filtered_amplitude(slide:slide+slide_window-1);
      %             tmp=sum(abs(tmp1-mean(tmp1)))/slide_window;
      
      
      if isempty(begin)&&tmp_amp>0.45*std(abs(diff_amp))%&&tmp_phase>0.2*std(abs(diff_phase)) %bend1,3:2.5 3 Thresh_hold
          begin=slide;
          s=[s; begin];
%                   hold on; plot([slide slide],[0,1],'r')
      end
      if ~isempty(begin)&&tmp_amp<std(abs(diff_amp))
          over=slide;
          begin=[];
          e=[e; over];
%                   hold on; plot([slide slide],[0,1],'g')
      end
  end
if isempty(s)||isempty(e)
    s=1;e=size(filtered_amplitude,1);disp([num2str(i),' lost']);
end
segment=[s(1) e(end)]; width=e(end)-s(1)+1; height=max(filtered_amplitude(:));
rectangle('Position',[s(1),0,width,height],'EdgeColor','m');
%%%%  [~,locs]= findpeaks(med_time,'NPeaks',10);segment=[locs(1) locs(end)]; width=locs(end)-locs(1)+1; height=max(filtered_amplitude(:));
%%%% rectangle('Position',[locs(1),0,width,height],'EdgeColor','m');
saveas(gcf,strcat(resultSavePath,'\',fileName,'.jpg'));
 end
  
  
%   figure; plot(filtered_amplitude);
%   hold on; plot([s s],[1,70],'r')
%   hold on; plot([e e],[1,70],'g')
%   
%   
%   s_final=s(basic(i,j));    % 给定第一个基准
%   for point_s=2:length(s)
%       if s(point_s)-s_final(end)>start_bet(i,j)
%           s_final=[s_final;s(point_s)];
%       end
%   end
%   s = s_final;
%   
%   e(find(diff(e)<150))=[];e_final=[];point_e_strat=1; %bend1,3:150
%   for point_s=1:length(s_final)
%       for point_e=point_e_strat:length(e)
%           if e(point_e)-s_final(point_s)>len_es(i,j)  %bend1:200;bend3:1000
%               e_final=[e_final;e(point_e)];
%               point_e_strat=point_e+1;
%               break;
%           end
%       end
%   end
%   e = e_final;
%   
%   
%   figure; plot(filtered_amplitude);
%   hold on; plot([s_final s_final],[1,25],'r')
%   hold on; plot([e_final e_final],[1,25],'g')
%   
%   %         segment{i,j}=[s(1:min(length(s),length(e))) e(1:min(length(s),length(e)))];
%   
%   segment=[s(1:min(length(s),length(e))) e(1:min(length(s),length(e)))];


