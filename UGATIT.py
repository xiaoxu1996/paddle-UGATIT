import time, itertools
import numpy as np
import paddle.fluid as fluid
import paddle
import sys
from networks import *
from utils import *
from glob import glob
from paddle.fluid.dygraph import MSELoss, L1Loss, BCELoss
from data_reader import *

def clip_rho(net,vmin=0,vmax=1):
    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(fluid.layers.clip(param,vmin,vmax))


# 定义模型的类
class UGATIT(object) :
    # 初始化参数
    def __init__(self, args):
        self.args = args

        #模型的名字
        if self.args.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'
    #建立模型
    def build_model(self):
        #加载数据
        self.trainA_list = os.path.join('/home/aistudio/data', 'trainA.txt')
        self.trainB_list = os.path.join('/home/aistudio/data', 'trainB.txt')
        self.testA_list = os.path.join('/home/aistudio/data', 'testA.txt')
        self.testB_list = os.path.join('/home/aistudio/data', 'testB.txt')

        self.trainA_reader = fluid.io.batch(data_reader(self.trainA_list, mode="TRAIN"), 
                                            batch_size=self.args.batch_size, drop_last=True)
        
        self.trainB_reader = fluid.io.batch(data_reader(self.trainB_list, mode="TRAIN"), 
                                            batch_size=self.args.batch_size, drop_last=True)

        self.testA_reader = fluid.io.batch(data_reader(self.testA_list, mode="TEST"), 
                                            batch_size=1, drop_last=False)

        self.testB_reader = fluid.io.batch(data_reader(self.testB_list, mode="TEST"), 
                                            batch_size=1, drop_last=False)

        #定义生成器，判别器
        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.args.ch, n_blocks=self.args.n_res, 
                                    img_size=self.args.img_size, light=self.args.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.args.ch, n_blocks=self.args.n_res, 
                                    img_size=self.args.img_size, light=self.args.light)
            self.disGA = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=7)
            self.disGB = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=7)
            self.disLA = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=5)
            self.disLB = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=5)

            #定义优化器
            self.G_optim = fluid.optimizer.AdamOptimizer(parameter_list=self.genA2B.parameters() + self.genB2A.parameters(), 
                                        learning_rate=self.args.lr, beta1=0.5, beta2=0.999, 
                                        regularization=fluid.regularizer.L2Decay(regularization_coeff=self.args.weight_decay))
            self.D_optim = fluid.optimizer.AdamOptimizer(parameter_list=self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters(), 
                                        learning_rate=self.args.lr, beta1=0.5, beta2=0.999,
                                        regularization=fluid.regularizer.L2Decay(regularization_coeff=self.args.weight_decay))
            self.MSELoss = MSELoss()
            self.L1Loss = L1Loss()
            #self.BCELoss = BCELoss()
        #self.Rho_clipper = RhoClipper(0, 1)

    #训练
    def train(self):
        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            start_iter = 1
            #加载模型
            if self.args.resume:
                model_list = glob(os.path.join(self.args.result_dir, self.args.dataset, 'model/', '*.pdparams'))
                if not len(model_list) == 0:
                    model_list.sort()
                    start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                    self.load(os.path.join(self.args.result_dir, self.args.dataset, 'model/'), start_iter)
                    print(" [*] Load SUCCESS")

            #开始训练
            print('training start !')
            start_time = time.time()
            for step in range(start_iter, self.args.iteration + 1):
                try:
                    real_A = next(trainA_iter)
                except:
                    trainA_iter = iter(self.trainA_reader())
                    real_A = next(trainA_iter)

                try:
                    real_B = next(trainB_iter)
                except:
                    trainB_iter = iter(self.trainB_reader())
                    real_B = next(trainB_iter)

                real_A = fluid.dygraph.to_variable(np.array(real_A))
                real_B = fluid.dygraph.to_variable(np.array(real_B))
                

                #更新判别器D
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)
                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                ones_1 = fluid.layers.ones_like(real_GA_logit)
                zeros_1 = fluid.layers.zeros_like(fake_GA_logit)
                ones_2 = fluid.layers.ones_like(real_GA_cam_logit)
                zeros_2 = fluid.layers.zeros_like(fake_GA_cam_logit)
                ones_3 = fluid.layers.ones_like(real_LA_logit)
                zeros_3 = fluid.layers.zeros_like(fake_LA_logit)
                ones_4 = fluid.layers.ones_like(real_LA_cam_logit)
                zeros_4 = fluid.layers.zeros_like(fake_LA_cam_logit)
                #计算损失
                D_ad_loss_GA = self.MSELoss(real_GA_logit, ones_1) + self.MSELoss(fake_GA_logit, zeros_1)
                D_ad_cam_loss_GA = self.MSELoss(real_GA_cam_logit, ones_2) + self.MSELoss(fake_GA_cam_logit, zeros_2)
                D_ad_loss_LA = self.MSELoss(real_LA_logit, ones_3) + self.MSELoss(fake_LA_logit, zeros_3)
                D_ad_cam_loss_LA = self.MSELoss(real_LA_cam_logit, ones_4) + self.MSELoss(fake_LA_cam_logit, zeros_4)
                D_ad_loss_GB = self.MSELoss(real_GB_logit, ones_1) + self.MSELoss(fake_GB_logit, zeros_1)
                D_ad_cam_loss_GB = self.MSELoss(real_GB_cam_logit, ones_2) + self.MSELoss(fake_GB_cam_logit, zeros_2)
                D_ad_loss_LB = self.MSELoss(real_LB_logit, ones_3) + self.MSELoss(fake_LB_logit, zeros_3)
                D_ad_cam_loss_LB = self.MSELoss(real_LB_cam_logit, ones_4) + self.MSELoss(fake_LB_cam_logit, zeros_4)

                D_loss_A = self.args.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.args.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_optim.minimize(Discriminator_loss)
                # self.disGA.clear_gradients()
                # self.disGB.clear_gradients()
                # self.disLA.clear_gradients()
                # self.disLB.clear_gradients()
                # self.genA2B.clear_gradients()
                # self.genB2A.clear_gradients()
                self.D_optim.clear_gradients()

                #更新生成器G
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = self.MSELoss(fake_GA_logit, ones_1)
                G_ad_cam_loss_GA = self.MSELoss(fake_GA_cam_logit, ones_2)
                G_ad_loss_LA = self.MSELoss(fake_LA_logit, ones_3)
                G_ad_cam_loss_LA = self.MSELoss(fake_LA_cam_logit, ones_4)
                G_ad_loss_GB = self.MSELoss(fake_GB_logit, ones_1)
                G_ad_cam_loss_GB = self.MSELoss(fake_GB_cam_logit, ones_2)
                G_ad_loss_LB = self.MSELoss(fake_LB_logit, ones_3)
                G_ad_cam_loss_LB = self.MSELoss(fake_LB_cam_logit, ones_4)

                G_recon_loss_A = self.L1Loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1Loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1Loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1Loss(fake_B2B, real_B)

                G_cam_loss_A = fluid.layers.mean(fluid.layers.sigmoid_cross_entropy_with_logits(fake_B2A_cam_logit, ones_2) + fluid.layers.sigmoid_cross_entropy_with_logits(fake_A2A_cam_logit, zeros_2))
                G_cam_loss_B = fluid.layers.mean(fluid.layers.sigmoid_cross_entropy_with_logits(fake_A2B_cam_logit, ones_2) + fluid.layers.sigmoid_cross_entropy_with_logits(fake_B2B_cam_logit, zeros_2))

                G_loss_A =  self.args.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.args.cycle_weight * G_recon_loss_A + self.args.identity_weight * G_identity_loss_A + self.args.cam_weight * G_cam_loss_A
                G_loss_B = self.args.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.args.cycle_weight * G_recon_loss_B + self.args.identity_weight * G_identity_loss_B + self.args.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.minimize(Generator_loss)
                self.D_optim.minimize(Discriminator_loss)
                # self.disGA.clear_gradients()
                # self.disGB.clear_gradients()
                # self.disLA.clear_gradients()
                # self.disLB.clear_gradients()
                # self.genA2B.clear_gradients()
                # self.genB2A.clear_gradients()
                self.G_optim.clear_gradients()
                    
                clip_rho(self.genA2B)
                clip_rho(self.genB2A)
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.args.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                if step % self.args.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.args.img_size * 7, 0, 3))
                    B2A = np.zeros((self.args.img_size * 7, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        try:
                            real_A = next(trainA_iter)
                        except:
                            trainA_iter = iter(self.trainA_reader())
                            real_A = next(trainA_iter)

                        try:
                            real_B = next(trainB_iter)
                        except:
                            trainB_iter = iter(self.trainB_reader())
                            real_B = next(trainB_iter)
                        
                        real_A = fluid.dygraph.to_variable(np.array(real_A))
                        real_B = fluid.dygraph.to_variable(np.array(real_B))

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
                        

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A = next(testA_iter)
                        except:
                            testA_iter = iter(self.testA_reader())
                            real_A = next(testA_iter)

                        try:
                            real_B = next(testB_iter)
                        except:
                            testB_iter = iter(self.testB_reader())
                            real_B = next(testB_iter)

                        real_A = fluid.dygraph.to_variable(np.array(real_A))
                        real_B = fluid.dygraph.to_variable(np.array(real_B))

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.args.save_freq == 0:
                    self.save(os.path.join(self.args.result_dir, self.args.dataset, 'model/'), step)

                # if step % 20 == 0:
                #     params = {}
                #     params['genA2B'] = self.genA2B.state_dict()
                #     params['genB2A'] = self.genB2A.state_dict()
                #     params['disGA'] = self.disGA.state_dict()
                #     params['disGB'] = self.disGB.state_dict()
                #     params['disLA'] = self.disLA.state_dict()
                #     params['disLB'] = self.disLB.state_dict()
                #     model_path = os.path.join(self.args.result_dir, self.args.dataset)
                #     fluid.save_dygraph(params, model_path + '_params_latest')
                #     # model_path = os.path.join(self.args.result_dir, self.args.dataset)
                #     # fluid.save_dygraph(self.genA2B.state_dict(), model_path + 'genA2B')


    def save(self, dir, step):
        model_path = dir
        fluid.save_dygraph(self.genA2B.state_dict(), model_path + '_genA2B_%07d' % step)
        fluid.save_dygraph(self.genB2A.state_dict(), model_path + '_genB2A_%07d' % step)
        fluid.save_dygraph(self.disGA.state_dict(), model_path + '_disGA_%07d' % step)
        fluid.save_dygraph(self.disGB.state_dict(), model_path + '_disGB_%07d' % step)
        fluid.save_dygraph(self.disLA.state_dict(), model_path + '_disLA_%07d' % step)
        fluid.save_dygraph(self.disLB.state_dict(), model_path + '_disLB_%07d' % step)

    def load(self, dir, step):
        load_path = dir
        param_genA2B, _ = fluid.load_dygraph(load_path + '_genA2B_%07d' % step)
        param_genB2A, _ = fluid.load_dygraph(load_path + '_genB2A_%07d' % step)
        param_disGA, _ = fluid.load_dygraph(load_path + '_disGA_%07d' % step)
        param_disGB, _ = fluid.load_dygraph(load_path + '_disGB_%07d' % step)
        param_disLA, _ = fluid.load_dygraph(load_path + '_disLA_%07d' % step)
        param_disLB, _ = fluid.load_dygraph(load_path + '_disLB_%07d' % step)

        self.genA2B.load_dict(param_genA2B)
        self.genB2A.load_dict(param_genB2A)
        self.disGA.load_dict(param_disGA)
        self.disGB.load_dict(param_disGB)
        self.disLA.load_dict(param_disLA)
        self.disLB.load_dict(param_disLB)

    def test(self):
        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            model_list = glob(os.path.join(self.args.result_dir, self.args.dataset, 'model/', '*.pdparams'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.args.result_dir, self.args.dataset, 'model/'), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, real_A in enumerate(self.testA_reader()):
                real_A = fluid.dygraph.to_variable(np.array(real_A))

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

                cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

            for n, real_B in enumerate(self.testB_reader()):
                real_B = fluid.dygraph.to_variable(np.array(real_B))

                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

                cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)


