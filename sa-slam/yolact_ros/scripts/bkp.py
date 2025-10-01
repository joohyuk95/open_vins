def prep_outlier_display(self, classesc, scoresc, boxesc, masksc, img, class_color=False, mask_alpha=0.45, fps_str=''):

      img_gpu = img / 255.0

      classes = classesc.copy()
      scores = scoresc.copy()
      boxes = boxesc.copy()
      masks = masksc.cpu().numpy()

      num_dets_to_consider = min(self.top_k, classes.shape[0])
      for j in range(num_dets_to_consider):
        if not (classes[j] == 0 or classes[j] == 5 or classes[j] == 28):
           np.delete(classes, j)
           np.delete(scores, j)
           np.delete(boxes, j)
           np.delete(masks, j)
           j = j-1
           num_dets_to_consider = num_dets_to_consider-1

      for j in range(num_dets_to_consider):
          if scores[j] < self.score_threshold:
              num_dets_to_consider = j
              break
          
          #return ((masks[j]) * 255).byte().cpu().numpy()


      # First, draw the masks on the GPU where we can do it really fast
      # Beware: very fast but possibly unintelligible mask-drawing code ahead
      # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
      if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
          # After this, mask is of size [num_dets, h, w, 1]
          masks = masks[:num_dets_to_consider, :, :, None]
          # This is 1 everywhere except for 1-mask_alpha where the mask is
          inv_alph_masks = masks * (-mask_alpha) + 1
          
          #img_gpu = img_gpu * inv_alph_masks.prod(dim=0) #+ masks_color_summand
          img_gpu = inv_alph_masks.prod(dim=0)

      # Then draw the stuff that needs to be done on the cpu
      # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
      img_numpy = (img_gpu * 255).byte().cpu().numpy()
      if num_dets_to_consider == 0:
          zero_img = np.zeros([img_numpy.shape[0],img_numpy.shape[1],1],dtype=np.uint8)
          zero_img.fill(0) # or img[:] = 255
          #print(num_dets_to_consider)
          return zero_img

      mask = np.uint8(img_numpy>200)
      
      img_numpy = np.invert(cv2.bitwise_and(img_numpy, img_numpy, mask=mask))

      # Inflate the mask a bit
      kernel_size = 5
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
      img_numpy = cv2.dilate(img_numpy, kernel, iterations=1)
      print(img_numpy.shape)
      print(self.depth_image.shape)
      return img_numpy