from nets.model import get_model
from dataset import S1P1, TEMPORAL
from nets.unet_model import UNet
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import DataUtil
from tensorboardX import SummaryWriter
import random
#from torchcontrib.optim import SWA
import datetime
import time


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,  best_train_loss, best_test_loss):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0

    epoch_size /= train_batch

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= int(epoch_size)+1:
                break
            imgs,boxes,labels = batch[0], batch[1], batch[2]
            imgs = imgs.view(-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
            boxes = torch.Tensor(boxes)
            boxes = boxes.view(1,-1)
            
            with torch.no_grad():
                if cuda:
                    imgs = Variable((imgs).type(torch.FloatTensor)).cuda()
                    boxes = [Variable((box).type(torch.FloatTensor)).cuda() for box in boxes]
                    labels = [Variable((label).type(torch.FloatTensor)).cuda() for label in labels]
                else:
                    imgs = Variable((imgs).type(torch.FloatTensor))
                    boxes = [Variable((box).type(torch.FloatTensor)) for box in boxes]
                    labels = [Variable((label).type(torch.FloatTensor)) for label in labels]
            
            losses = train_util.train_step(imgs, boxes, labels)
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
            total_loss += total
            rpn_loc_loss += rpn_loc
            rpn_cls_loss += rpn_cls
            roi_loc_loss += roi_loc
            roi_cls_loss += roi_cls
            
            pbar.set_postfix(**{'total'    : total_loss.item() / (iteration + 1), 
                                'rpn_loc'  : rpn_loc_loss.item() / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss.item() / (iteration + 1), 
                                'roi_loc'  : roi_loc_loss.item() / (iteration + 1), 
                                'roi_cls'  : roi_cls_loss.item() / (iteration + 1), 
                                'lr'       : get_lr(optimizer)})
            pbar.update(1)
    writer.add_scalar('train_loss', total_loss/(epoch_size+1), epoch)
    writer.add_scalar('train_roi_cls_loss', roi_cls_loss/len(gen),epoch)
    writer.add_scalar('train_roi_loc_loss', roi_loc_loss/len(gen),epoch)
    writer.add_scalar('train_rpn_cls_loss', rpn_cls_loss/len(gen),epoch)
    writer.add_scalar('train_rpn_loc_loss', rpn_loc_loss/len(gen),epoch)
    writer.add_scalar('train_lr', get_lr(optimizer),epoch)

    print('Start Validation')

    epoch_size_val /= test_bacth
    
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= int(epoch_size_val)+1:
                break
            imgs,boxes,labels = batch[0], batch[1], batch[2]
            imgs = imgs.view(-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
            boxes = torch.Tensor(boxes)
            boxes = boxes.view(1,-1)
            with torch.no_grad():
                if cuda:
                    imgs = Variable((imgs).type(torch.FloatTensor)).cuda()
                    boxes = [Variable((box).type(torch.FloatTensor)).cuda() for box in boxes]
                    labels = [Variable((label).type(torch.FloatTensor)).cuda() for label in labels]
                else:
                    imgs = Variable((imgs).type(torch.FloatTensor))
                    boxes = [Variable((box).type(torch.FloatTensor)) for box in boxes]
                    labels = [Variable((label).type(torch.FloatTensor)) for label in labels]

                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels)
                rpn_loc, rpn_cls, roi_loc, roi_cls, val_total = losses
                val_toal_loss += val_total
                rpn_loc_loss += rpn_loc
                rpn_cls_loss += rpn_cls
                roi_loc_loss += roi_loc
                roi_cls_loss += roi_cls
            pbar.set_postfix(**{'total_loss': val_toal_loss.item() / (iteration + 1)})
            pbar.update(1)
    writer.add_scalar('test_loss', val_toal_loss/(epoch_size_val+1), epoch)
    writer.add_scalar('test_roi_cls_loss', roi_cls_loss/len(genval),epoch)
    writer.add_scalar('test_roi_loc_loss', roi_loc_loss/len(genval),epoch)
    writer.add_scalar('test_rpn_cls_loss', rpn_cls_loss/len(genval),epoch)
    writer.add_scalar('test_rpn_loc_loss', rpn_loc_loss/len(genval),epoch)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    best_train_loss = min(best_train_loss, total_loss/(epoch_size+1))
    best_test_loss = min(best_test_loss, val_toal_loss/(epoch_size_val+1))
    checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch+1,
            "best_test_loss":  best_test_loss,
            "best_train_loss": best_train_loss,
            'lr_schedule': lr_scheduler.state_dict()
        }
    
    if (epoch+1) % 5 == 0:
        torch.save(checkpoint, 'logs/'+log_name+'/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))
    
if __name__ == "__main__":
    Kaggle = True

    # 设置训练的数据集
    dataset = TEMPORAL
   
    # 设置实验名
    log_name = "16-torch-vgg-S0"
    
    # 设置主干特征提取网络类型
    BACKBONE = "vgg"

    # 设置是否断点训练
    RESUME = False
    path_checkpoint = "logs/13-ori-rpnNms1-clsDrop03-192S1ALL/Epoch109-Total_Loss0.6752-Val_Loss19.3184.pth"

    # 初始化数据集参数
    dataset_name = dataset.name
    NUM_CLASSES = dataset.num_classes
    train_batch = dataset.train_batch
    test_bacth = dataset.test_batch
    ANCHOR = dataset.anchor
    IMAGE_SHAPE = dataset.image_shape
    if Kaggle is True:
        DataUtil.home_dir = dataset.kaggle_dir

    # 初始化随机数种子
    setup_seed(510)
    # 设置loss绘制日志保存路径
    log_name += "-"+dataset_name
    writer = SummaryWriter('logs/'+log_name+'/'+str(datetime.date.today())+"_"+str(time.time())[-5:])

    # 初始化网络结构
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if Cuda else "cpu")
    print("use" , device)
    model = get_model(dataset, BACKBONE).to(device)

    # 训练模式
    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        net = net.cuda()
    cudnn.benchmark = True
    
    # 加载数据集
    num_train, gen = DataUtil.get_data_loader(dataset_name,"train",train_batch,True)
    num_val, gen_val = DataUtil.get_data_loader(dataset_name,"test",test_bacth,False)

    # 初始化最佳loss为一个极大值
    best_train_loss = 1000
    best_test_loss = 1000

    if True:
        # 设置学习率
        lr = 3e-4
        # 设置起始epoch
        Freeze_Epoch = 0
        # 设置结束epoch
        Unfreeze_Epoch = 260

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        #optimizer = SWA(optimizer)#, swa_start=10, swa_freq=5, swa_lr=1e-2)	
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.99)

        # 恢复断点
        if RESUME:
            checkpoint = torch.load(path_checkpoint,map_location=device)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            Freeze_Epoch = checkpoint['epoch']  # 设置开始的epoch

            best_test_loss = checkpoint['best_test_loss']
            best_train_loss = checkpoint['best_train_loss']
            lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
     
        epoch_size = num_train
        epoch_size_val = num_val

        train_util = FasterRCNNTrainer(model,optimizer)

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda, best_train_loss, best_test_loss)
            lr_scheduler.step()
        #     if (epoch+1) >= 10 and (epoch+1) % 5 == 0:
        #         optimizer.update_swa()
        # optimizer.swap_swa_sgd()

    writer.close()


    from utils.utils import bbox_iou, detection_acc
    import os
    import re
    from tensorboardX import SummaryWriter
    
    # log保存路径
    writer = SummaryWriter('logs_map/'+str(datetime.date.today())+"_"+log_name+"_"+str(time.time())[-5:])
    # 模型保存文件夹
    dirs = 'logs/'+log_name
    # 训练集还是测试集
    data_type = "test"
    # 是否绘制mAP
    MAP = False            

    actions = dataset.actions
    '''
    加载测试集
    '''
    num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name,data_type,1,False)

    fileList = os.listdir(dirs)
    regex = re.compile(r'\d+')
    fileList.sort(key = lambda x: int(regex.findall(x)[0]))

    fileList = fileList[0::1]

    for each in fileList:
        if len(each) < 3 or each[-3:] != "pth":
            continue
        
        epoch = int(regex.findall(each)[0])

        path_checkpoint = os.path.join(dirs, each)  # 断点路径

        #加载自定义的模型
        try:
            checkpoint = torch.load(path_checkpoint,  map_location=device)  # 加载断点
        except RuntimeError:
            print("RuntimeError")
            continue
            
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        if MAP:
            os.system("rm input/detection-results/result*.txt")
            os.system("rm input/ground-truth/result*.txt")

        model.eval()
        np.set_printoptions(suppress=True)
        ious_all = 0
        dete_all = 0.0
        final_all = 0.0
        acc = 0.0
        cnt = 0
        i = 0
        for (data, bbox, label) in tqdm(test_data_loader):
            with torch.no_grad():
                data = data.reshape([-1,IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
                if Cuda:
                    dataV = Variable(data).cuda()
                    bboxV = Variable(bbox).cuda()
                    labelV = Variable(label).cuda()
                else:
                    dataV = Variable(data)
                    bboxV = Variable(bbox)
                    labelV = Variable(label)

                predictions = model(dataV.unsqueeze(3))
                if (predictions[1][0]['boxes'].shape[0]) == 0:
                    continue
                if len(predictions[1])>1:
                    print(len(predictions[1]))
                bbox = predictions[1][0]["boxes"][:,1:4:2].view(-1,2)
                conf = predictions[1][0]["scores"]
                label = predictions[1][0]["labels"]
                idx = 0
                max_iou = bbox_iou(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV.tolist()))[0][0]
                dete_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV.tolist()))[0][0]
                final_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV.tolist()), int(label[idx])==int(labelV[0]))[0][0]
                i+=1
                if MAP:
                    resultFile = open("input/detection-results/result"+str(i)+".txt", "w")
                    #for j in range(0, bbox.shape[0]):
                    j = idx
                    resultFile.write("%s %.6f %f %f %f %f\n"%(actions[int(label[j])], conf[j], bbox[j][0], 0, bbox[j][1], 90))
                    resultFile.close()
                    groundFile = open("input/ground-truth/result"+str(i)+".txt", "w")
                    for j in range(0, bboxV.shape[0]):
                        groundFile.write("%s %f %f %f %f\n"%(actions[int(labelV[j][0])], bboxV[j][0], 0, bboxV[j][1], 90))
                    groundFile.close()

                if int(label[idx])==int(labelV[0]):
                    acc = acc + 1
                ious_all += max_iou
                dete_all += dete_acc
                final_all += final_acc
                cnt+=1

        print("Epoch:",epoch," 有效预测：",cnt)
        if cnt == 0:
            continue
        
        dete_all /= num_test_instances
        final_all /= num_test_instances
        ious_all /= num_test_instances
        acc /= num_test_instances
        print("IOU: ",ious_all)
        print("分类准确度：", acc)
        print("检测精度: ",dete_all)
        print("检测分类精度: ",final_all)
        writer.add_scalar(data_type+"_IOU", ious_all, epoch)
        writer.add_scalar(data_type+"_detection", dete_all, epoch)
        writer.add_scalar(data_type+"_final", final_all, epoch)
        writer.add_scalar(data_type+'_ACC', acc, epoch)
        writer.add_scalar(data_type+'_CNT', cnt, epoch)
    writer.close()
