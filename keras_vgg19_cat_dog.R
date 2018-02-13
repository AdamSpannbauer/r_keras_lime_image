library(keras)

#####################
# helper functions
#####################
# define image preprocessor for use with keras vgg19
image_preprocessor = function(image_path) {
  image_load(image_path, target_size = c(224,224)) %>% 
    image_to_array() %>% 
    array_reshape(c(1, dim(.))) %>% 
    imagenet_preprocess_input()
}

#function to read in files housing all cat/dog breeds in imagenet labels
read_dog_cat_labels = function(path) {
  labs = readLines(path)
  labs = trimws(unlist(strsplit(labs, ',')))
  labs = gsub('\\s+', '_', labs)
  return(labs)
}
#------------------------------------------------

#####################
# prep images for classification
#####################
# define image path to classify
image_paths = list.files('images', 
                         recursive = TRUE, 
                         full.names = TRUE)

# preprocess images
image_list = lapply(image_paths, image_preprocessor)
#------------------------------------------------

#####################
# load model and make predictions
#####################
# load vgg19 model pretrained with imagenet
model = application_vgg19()

# get model prediction
preds = lapply(image_list, function(i) {
  imagenet_decode_predictions(predict(model, i), top = 1)[[1]]
})
#------------------------------------------------

#####################
# convert labels to 'Cat', 'Dog', 'NoCatDog'
#####################
dog_labs = read_dog_cat_labels('data/dog_classes.txt')
cat_labs = read_dog_cat_labels('data/cat_classes.txt')

#convert all dog / cat breeds to just 'dog' and 'cat'
preds = lapply(preds, function(df_i) {
  df_i$class_description[df_i$class_description %in% dog_labs] = 'Dog'
  df_i$class_description[df_i$class_description %in% cat_labs] = 'Cat'
  df_i$class_description[df_i$class_description != 'Dog' &
                           df_i$class_description != 'Cat'] = 'NoCatDog'
  df_i
})
#------------------------------------------------

#####################
# pretty up output
#####################
pred_df = do.call(rbind, preds)
pred_df$class_name = NULL
pred_df$file_name = unlist(image_paths)

#   class_description     score                                  file_name
# 1               Cat 0.4494216            images/cats/goober_lounging.JPG
# 2               Cat 0.4777499           images/cats/google_tabby_cat.jpg
# 3               Dog 0.3038349       images/cats/jasper_announce_sign.jpg
# 4               Cat 0.1475925                images/cats/lilly_perch.JPG
# 5               Dog 0.5802087        images/dogs/tonks_announce_sign.jpg
# 6               Dog 0.7851309 images/dogs/tonks_jasper_announce_sign.jpg
# 7               Dog 0.9420919        images/dogs/tonks_proposal_sign.jpg
#------------------------------------------------