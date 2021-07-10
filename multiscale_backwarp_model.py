import getopt
import math
import sys
import torch
from torch import nn as nn
from functools import partial
import module_util as mutil

def default_conv(in_channels, out_channels, kernel_size, bias=True):
	return nn.Conv2d(
		in_channels, out_channels, kernel_size,
		padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
	def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(conv(n_feat, 4 * n_feat, 3, bias))
				m.append(nn.PixelShuffle(2))
				if bn: m.append(nn.BatchNorm2d(n_feat))
				if act: m.append(act())
		elif scale == 3:
			m.append(conv(n_feat, 9 * n_feat, 3, bias))
			m.append(nn.PixelShuffle(3))
			if bn: m.append(nn.BatchNorm2d(n_feat))
			if act: m.append(act())
		else:
			raise NotImplementedError

		super(Upsampler, self).__init__(*m)


class LateralBlock(nn.Module):
	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)
		if ch_in != ch_out:
			self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

	def forward(self, x):
		fx = self.f(x)
		if fx.shape[1] != x.shape[1]:
			x = self.conv(x)

		return fx + x


class DownSamplingBlock(nn.Module):

	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)


	def forward(self, x):
		return self.f(x)


class UpSamplingBlock(nn.Module):

	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			# nn.UpsamplingNearest2d(scale_factor = 2),
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)

	def forward(self, x):
		return self.f(x)

try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################
torch.set_grad_enabled(True) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use


class pwcnet(torch.nn.Module):
	def __init__(self):
		super(pwcnet, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
		# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end
			def backwarp(self,tenInput, tenFlow):

				tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1).type_as(tenInput)
				tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3]).type_as(tenInput)

				backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1)
				# end


				backwarp_tenPartial = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
				# end

				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

				tenInput = torch.cat([ tenInput, backwarp_tenPartial ], 1)


				tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

				tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

				return tenOutput[:, :-1, :, :] * tenMask

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=self.backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				# end

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
		# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.netMain(tenInput)
		# end
		# end

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		#[1, 3, 64, 112]
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)

		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])






# --- Feature pyramid extractor for multiple features--- #
class Feature_pyramid_extractor(torch.nn.Module):
	def __init__(self):

		super(Feature_pyramid_extractor, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=32,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.PReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.PReLU(),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.PReLU(),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(64, 96, 3, 2, 1),
			nn.PReLU(),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(96, 96, 3, 1, 1),
			nn.PReLU(),
		)


	def forward(self, frame1, frame3):
		'''
        fpn:feature_left_1 torch.Size([1, 32, 256, 448])
        '''
		#extract features from frame1
		x_left_1 = self.conv1(frame1)
		x_left_2 = self.conv2(x_left_1)
		x_left_3=self.conv3(x_left_2)
		x_left_4=self.conv4(x_left_3)
		x_left_5=self.conv5(x_left_4)
		x_left_6=self.conv6(x_left_5)
		feature_left_1=x_left_2
		feature_left_2=x_left_4
		feature_left_3=x_left_6

		# extract features from frame3
		x_right_1 = self.conv1(frame3)
		x_right_2 = self.conv2(x_right_1)
		x_right_3 = self.conv3(x_right_2)
		x_right_4 = self.conv4(x_right_3)
		x_right_5 = self.conv5(x_right_4)
		x_right_6 = self.conv6(x_right_5)
		feature_right_1 = x_right_2
		feature_right_2 = x_right_4
		feature_right_3 = x_right_6
		'''
        feature_right_1: cuda:0
        '''

		return feature_left_1,feature_left_2,feature_left_3, \
			   feature_right_1,feature_right_2,feature_right_3


class warp(torch.nn.Module):
	def __init__(self):
		super(warp, self).__init__()
		self.get_flow = pwcnet()
		#load in pre_trained_optical_weights
		pretrained_dict=torch.load('pwc_interpolation.pkl')
		model_dict=self.get_flow.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		self.get_flow.load_state_dict(model_dict)

		self.get_feature = Feature_pyramid_extractor()


	def three_size_for_backwarp(self,input1,input2):
		batch_size = input1.shape[0]
		frame_width = input1.shape[3]
		frame_height = input1.shape[2]


		tenPreprocessedFirst = input1.view(batch_size, 3, frame_height, frame_width)#[1, 3, 64, 112]
		tenPreprocessedSecond = input2.view(batch_size, 3, frame_height, frame_width)#[1, 3, 64, 112]

		l3_flow=self.get_flow(tenPreprocessedFirst, tenPreprocessedSecond)
		l2_flow=2 *torch.nn.functional.upsample(l3_flow, scale_factor=2, mode='bilinear', align_corners=False)
		l1_flow =2 *torch.nn.functional.upsample(l2_flow, scale_factor=2, mode='bilinear', align_corners=False)

		return l1_flow,l2_flow,l3_flow

	def backwarp(self, tenInput, tenFlow):
		"""tenFlow is optical-flow import from pwc-net
		   tenInput is one input original images
		   use two images to generate optical-flow and warping base on one original imput image
		"""
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
			1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1).type_as(tenInput)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
			1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3]).type_as(tenInput)

		backwarp_tenGrid = torch.cat([tenHor, tenVer], 1)

		tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
							 tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
		return torch.nn.functional.grid_sample(input=tenInput ,
											   grid=(backwarp_tenGrid+ tenFlow).permute(0, 2, 3, 1) ,
											   mode='bilinear', padding_mode='zeros', align_corners=False)


	def forward(self,frame1, frame3):
		# generate three level flow (64,32,16)
		flow1_big=self.three_size_for_backwarp(frame1, frame3)[0]
		flow2_big=self.three_size_for_backwarp(frame3, frame1)[0]
		flow1_middle=self.three_size_for_backwarp(frame1, frame3)[1]
		flow2_middle=self.three_size_for_backwarp(frame3, frame1)[1]
		flow1_small=self.three_size_for_backwarp(frame1, frame3)[2]
		flow2_small=self.three_size_for_backwarp(frame3, frame1)[2]
		# get feature (64,32,16)
		feature_left_1, feature_left_2, feature_left_3, feature_right_1, feature_right_2, feature_right_3 = self.get_feature(frame1, frame3)

		feature_forward = self.backwarp(tenInput=feature_left_1, tenFlow=flow1_big)
		pic_forward = self.backwarp(tenInput=frame1, tenFlow=flow1_big)
		pic_backward = self.backwarp(tenInput=frame3, tenFlow=flow2_big )
		feature_backward = self.backwarp(tenInput=feature_right_1, tenFlow=flow2_big)

		#second-level
		feature_second_forward = self.backwarp(tenInput=feature_left_2, tenFlow=flow1_middle)
		feature_second_backward = self.backwarp(tenInput=feature_right_2, tenFlow=flow2_middle)

		#third-level
		feature_third_forward = self.backwarp(tenInput=feature_left_3, tenFlow=flow1_small)
		feature_third_backward = self.backwarp(tenInput=feature_right_3, tenFlow=flow2_small)
		first_level = torch.cat((feature_forward, pic_forward, pic_backward, feature_backward), 1)
		second_level = torch.cat((feature_second_forward, feature_second_backward), 1)
		third_level = torch.cat((feature_third_forward, feature_third_backward), 1)

		return first_level, second_level, third_level


class RC_CALayer(nn.Module):
	def __init__(self, channel, reduction=8):
		super(RC_CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
			nn.ReLU(),
			nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
	def __init__(
			self, conv, n_feat, kernel_size, reduction,
			bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

		super(RCAB, self).__init__()
		modules_body = []
		for i in range(2):
			modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
			if bn: modules_body.append(nn.BatchNorm2d(n_feat))
			if i == 0: modules_body.append(act)
		modules_body.append(RC_CALayer(n_feat, reduction))
		self.body = nn.Sequential(*modules_body)
		self.res_scale = res_scale

	def forward(self, x):
		res = self.body(x)
		res = res+ x
		return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
	def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
		super(ResidualGroup, self).__init__()
		modules_body = []
		modules_body = [
			RCAB(
				conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
			for _ in range(n_resblocks)]
		modules_body.append(conv(n_feat, n_feat, kernel_size))
		self.body = nn.Sequential(*modules_body)

	def forward(self, x):
		res = self.body(x)
		res = res+ x
		return res


class rcan(nn.Module):
	def __init__(self, conv=default_conv):
		super(rcan, self).__init__()
		n_resgroups = 5
		n_resblocks = 10
		n_feats = 32
		kernel_size = 3
		reduction = 8
		act = nn.ReLU(True)

		modules_head = [conv(3, n_feats, kernel_size)]

		# define body module
		modules_body = [
			ResidualGroup(
				conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
			for _ in range(n_resgroups)]

		modules_body.append(conv(n_feats, n_feats, kernel_size))

		# define tail module
		modules_tail = [
			Upsampler(conv, 4, n_feats, act=False),
			conv(n_feats, 3, kernel_size)]

		self.head = nn.Sequential(*modules_head)
		self.body = nn.Sequential(*modules_body)
		self.tail = nn.Sequential(*modules_tail)

	def forward(self, x):
		x_feat = self.head(x)
		res = self.body(x_feat)
		out_feat = res + x_feat
		out_feat= self.tail(out_feat)
		return out_feat


class video_interpoaltion(torch.nn.Module):
	def __init__(self, in_chs_1=70, in_chs_2=128, in_chs_3= 192, out_chs=3, grid_chs=[32, 64, 96]):
		super().__init__()

		#refine
		self.get_input = warp()
		self.n_row = 3
		self.n_col = 6
		self.n_chs = grid_chs
		assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

		self.lateral_init_1 = LateralBlock(in_chs_1, self.n_chs[0])
		self.lateral_init_2 = LateralBlock(in_chs_2, self.n_chs[1])
		self.lateral_init_3 = LateralBlock(in_chs_3, self.n_chs[2])


		for r, n_ch in enumerate(self.n_chs):
			for c in range(self.n_col - 1):
				setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

		for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
			for c in range(int(self.n_col / 2)):
				setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

		for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
			for c in range(int(self.n_col / 2)):
				setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

		self.lateral_final = LateralBlock(self.n_chs[0], 3)

	def forward(self, frame1,frame3):

		first_level, second_level, third_level = self.get_input(frame1, frame3)


		state_00 = self.lateral_init_1(first_level)
		state_10 = self.lateral_init_2(second_level)
		state_20 = self.lateral_init_3(third_level)

		state_01 = self.lateral_0_0(state_00)
		state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
		state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

		state_02 = self.lateral_0_1(state_01)
		state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
		state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

		state_23 = self.lateral_2_2(state_22)
		state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
		state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

		state_24 = self.lateral_2_3(state_23)
		state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
		state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

		state_25 = self.lateral_2_4(state_24)
		state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
		state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)
		LR_output= self.lateral_final(state_05)

		return LR_output


############### extract 3 frame features ###############
class feature_extraction(nn.Module):
	def __init__(self, nf=64, front_RBs=5):
		super(feature_extraction, self).__init__()
		self.nf = nf
		ResidualBlock_noBN_f = partial(mutil.ResidualBlock_noBN, nf=nf)

		self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
		self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)
		self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
		self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
		self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
	def forward(self, frame1,interpolated_frame,frame3):
		frame1_feature = self.lrelu(self.conv_first(frame1))
		frame1_feature = self.feature_extraction(frame1_feature)

		interpolated_frame_feature = self.lrelu(self.conv_first(interpolated_frame))
		interpolated_frame_feature = self.feature_extraction(interpolated_frame_feature)

		frame3_feature = self.lrelu(self.conv_first(frame3))
		frame3_feature = self.feature_extraction(frame3_feature)

		return frame1_feature,interpolated_frame_feature,frame3_feature


############### use 3 frame features to enhance the interpolated frame features ###############
class TSAFusion(nn.Module):
	"""Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

	def __init__(self, num_feat=64, num_frame=3, center_frame_idx=1):
		super(TSAFusion, self).__init__()
		self.center_frame_idx = center_frame_idx
		# temporal attention (before fusion conv)
		self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
		self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
		self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

		# spatial attention (after fusion conv)
		self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
		self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
		self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
		self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
		self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
		self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
		self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
		self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
		self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
		self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
		self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
		self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.upsample = nn.Upsample(
			scale_factor=2, mode='bilinear', align_corners=False)

	def forward(self, aligned_feat):
		"""
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
		b, t, c, h, w = aligned_feat.size()
		# temporal attention
		embedding_ref = self.temporal_attn1(
			aligned_feat[:, self.center_frame_idx, :, :, :].clone())
		embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
		embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

		corr_l = []  # correlation list
		for i in range(t):
			emb_neighbor = embedding[:, i, :, :, :]
			corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
			corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
		corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
		corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
		corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
		aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

		# fusion
		feat = self.lrelu(self.feat_fusion(aligned_feat))

		# spatial attention
		attn = self.lrelu(self.spatial_attn1(aligned_feat))
		attn_max = self.max_pool(attn)
		attn_avg = self.avg_pool(attn)
		attn = self.lrelu(
			self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
		# pyramid levels
		attn_level = self.lrelu(self.spatial_attn_l1(attn))
		attn_max = self.max_pool(attn_level)
		attn_avg = self.avg_pool(attn_level)
		attn_level = self.lrelu(
			self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
		attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
		attn_level = self.upsample(attn_level)

		attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
		attn = self.lrelu(self.spatial_attn4(attn))
		attn = self.upsample(attn)
		attn = self.spatial_attn5(attn)
		attn_add = self.spatial_attn_add2(
			self.lrelu(self.spatial_attn_add1(attn)))
		attn = torch.sigmoid(attn)

		# after initialization, * 2 makes (attn * 2) to be close to 1.
		enhanced_feature = feat * attn * 2 + attn_add
		return enhanced_feature


############# reconstruction and upsampling ###################
class HR_tail(nn.Module):
	def __init__(self, nf=64, back_RBs=10):
		super(HR_tail, self).__init__()
		ResidualBlock_noBN_f = partial(mutil.ResidualBlock_noBN, nf=nf)
		#### reconstruction
		self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)
		#### upsampling
		self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
		self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
		self.pixel_shuffle = nn.PixelShuffle(2)
		self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
		self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
	def forward(self, feats):
		B, T, C, H, W = feats.size()
		feats = feats.view(B*T, C, H, W)
		out = self.recon_trunk(feats)
		out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
		out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

		out = self.lrelu(self.HRconv(out))
		out = self.conv_last(out)
		_, _, K, G = out.size()
		outs = out.view(B, T, -1, K, G)
		return outs


class VISRNet(nn.Module):
	def __init__(self):
		super(VISRNet, self).__init__()
		self.interpolate_middle_frame=video_interpoaltion()
		self.all_frame_feature_extraction=feature_extraction()
		self.middle_frame_enhancement=TSAFusion()
		self.super_resolution=HR_tail()

	def forward(self, frame1, frame3):
		interpolated_frame=self.interpolate_middle_frame(frame1, frame3)
		frame1_feature,interpolated_frame_feature,frame3_feature=self.all_frame_feature_extraction(frame1,interpolated_frame,frame3)
		stack_features=torch.stack((frame1_feature,interpolated_frame_feature,frame3_feature),dim=1,out=None)
		enhanced_features=self.middle_frame_enhancement(stack_features)
		stack_features2=torch.stack((frame1_feature,enhanced_features,frame3_feature),dim=1,out=None)
		all_frame_SR_output=self.super_resolution(stack_features2)
		SR_frame1,SR_middle_frame,SR_frame3=all_frame_SR_output[:,0,:],all_frame_SR_output[:,1,:],all_frame_SR_output[:,2,:]
		#(3,64,64)  (3,256,256)
		return interpolated_frame,SR_frame1,SR_middle_frame,SR_frame3

if __name__ == '__main__':
	frame1=torch.randn(2,3,64,64).cuda()
	frame3=torch.randn(2,3,64,64).cuda()
	net=VISRNet()
	net.cuda()
	interpolated_frame,SR_frame1,SR_middle_frame,SR_frame3=net(frame1,frame3)






