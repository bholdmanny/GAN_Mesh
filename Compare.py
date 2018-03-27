# -*-coding:utf-8 -*-
#!/usr/bin/python3
import os, cv2
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import scipy.spatial
import facenet

class App:
	def __init__(self, modeldir, image_size =200):
		self.image_size = image_size
		with tf.Graph().as_default():
			config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0), log_device_placement=False)
			self.sess = tf.Session(config = config)
			facenet.load_model(os.path.join(modeldir, "20170512-110547/20170512-110547.pb"))
			self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			self.embedding_size = self.embeddings.get_shape()[1]

	def Feature(self, imagedir):
		scaled_reshape = []
		image = cv2.imread(imagedir)
		image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
		image = np.multiply(np.subtract(image, np.mean(image)), 1/(np.maximum(np.std(image), 1.0/np.sqrt(image.size))))
		scaled_reshape.append(image.reshape(-1, self.image_size, self.image_size, 3))
		emb_array1 = np.zeros((1, self.embedding_size))
		emb_array1[0, :] = self.sess.run(self.embeddings, feed_dict={self.images_placeholder: scaled_reshape[0], self.phase_train_placeholder: False })[0]
		return emb_array1[0, :]

	def Compare(self, feature1, feature2):
		cos_distance = 1 - scipy.spatial.distance.cosine(feature1,feature2)
		cos_distance = float("%.3f" %abs(cos_distance))
		return cos_distance

	def Simi(self, pic1, pic2):
		feature1, feature2 = self.Feature(pic1), self.Feature(pic2)
		return self.Compare(feature1, feature2)

def main(*args, **kwargs):
	api = App("./models")
	img = os.listdir(args[0]); img.sort()
	N = len(img); ss = np.zeros((N,N))
	os.chdir(args[0])
	for i in range(N):
		for j in range(N):
			ss[i][j] = api.Simi(img[i],img[j])
	print(img); print(ss)
	return ss,img

if __name__ == '__main__':
	from sys import argv
	main(argv[1])
 