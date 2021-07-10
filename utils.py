import torch
def normalize_feature(input1, input2):
	jointly_input = (input1 + input2)/2
	feature_jointly_mean = jointly_input.view(1, 64, -1).mean(dim=2).view(1, 64, 1, 1)
	feature_jointly_std = jointly_input.view(1, 64, -1).std(dim=2).view(1, 64, 1, 1)
	feature1_norm = (input1 - feature_jointly_mean) / feature_jointly_std
	feature2_norm=(input2 - feature_jointly_mean) / feature_jointly_std
	return feature1_norm,feature2_norm

def normalize_pic(input1, input2):
	jointly_input = (input1 + input2) / 2
	img_jointly_mean = jointly_input.view(1, 3, -1).mean(dim=2).view(1, 3, 1, 1)
	img_jointly_std = jointly_input.view(1, 3, -1).std(dim=2).view(1, 3, 1, 1)
	input1_norm = (input1 - img_jointly_mean) / img_jointly_std
	input2_norm = (input2 - img_jointly_mean) / img_jointly_std
	return input1_norm, input2_norm, img_jointly_mean, img_jointly_std



def loop_feature(fature1,fature2,batch_size):
	first_fature1 = fature1[0:64, :, :, :]
	first_fature2 = fature2[0:64, :, :, :]
	feature_after_norm1, feature_after_norm3 = normalize_feature(first_fature1, first_fature2)[0], normalize_feature(first_fature1, first_fature2)[1]
	for i in range(batch_size-1):
		i = i + 1
		fature1_var = fature1[i*64:(i + 1)*64, :, :, :]
		fature2_var = fature2[i*64:(i + 1)*64, :, :, :]
		norm1_new, norm3_new =normalize_feature(fature1_var, fature2_var)[0], normalize_feature(fature1_var, fature2_var)[1]
		feature_concat1 = torch.cat((feature_after_norm1, norm1_new), 0)
		feature_concat3 = torch.cat((feature_after_norm3, norm3_new), 0)
		feature_after_norm1 = feature_concat1
		feature_after_norm3 = feature_concat3

	return feature_after_norm1,feature_after_norm3


def loop_pic(frame1,frame3,batch_size):
	first_frame1 = frame1[0:1, :, :, :]
	first_frame3 = frame3[0:1, :, :, :]
	after_norm1, after_norm3 = normalize_pic(first_frame1, first_frame3)[0], normalize_pic(first_frame1, first_frame3)[1]
	mean1, std1 = normalize_pic(first_frame1, first_frame3)[2], normalize_pic(first_frame1, first_frame3)[3]
	for i in range(batch_size-1):
		i = i + 1
		frame1_var = frame1[i:i + 1, :, :, :]
		frame3_var = frame3[i:i + 1, :, :, :]
		norm1_new, norm3_new =normalize_pic(frame1_var, frame3_var)[0], normalize_pic(frame1_var, frame3_var)[1]
		mean_new, std_new = normalize_pic(frame1_var, frame3_var)[2], normalize_pic(frame1_var, frame3_var)[3]
		concat1 = torch.cat((after_norm1, norm1_new), 0)
		concat3 = torch.cat((after_norm3, norm3_new), 0)
		concat_mean = torch.cat((mean1, mean_new), 0)
		concat_std = torch.cat((std1, std_new), 0)
		after_norm1 = concat1
		after_norm3 = concat3
		mean1 = concat_mean
		std1 = concat_std
	return after_norm1,after_norm3,mean1,std1