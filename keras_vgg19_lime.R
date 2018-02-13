# install image branch of lime
# devtools::install_github('thomasp85/lime',  ref = 'image')

library(keras)
library(lime)

# load vgg19 model pretrained with imagenet
model = application_vgg19()

# define image path to classify
image_path = 'images/dogs/tonks_proposal_sign.jpg'

# define image preprocessor for use with keras vgg19
image_preprocessor = function(image_path) {
  image_load(image_path, target_size = c(224,224)) %>% 
    image_to_array() %>% 
    array_reshape(c(1, dim(.))) %>% 
    imagenet_preprocess_input()
}

# preprocess image
image = image_preprocessor(image_path)

# get model prediction
preds = predict(model, image) %>% 
  imagenet_decode_predictions(top = 1)
# print(preds[[1]])
##  class_name class_description     score
##1  n02085620         Chihuahua 0.9420919

# add model type and and predict model methods to keras model
#class(model)
model_type.keras.engine.training.Model = function(x, ...) {
  return("classification")
}
predict_model.keras.engine.training.Model = function(x, newdata, type) {
  res = imagenet_decode_predictions(predict(x, newdata), top = 1)
  switch(
    type,
    raw = data.frame(Response = vapply(res, function(x) x$class_name, character(1)), stringsAsFactors = FALSE),
    prob = data.frame(Response = vapply(res, function(x) x$score, numeric(1)), stringsAsFactors = FALSE)
  )
}

#create image explainer (call imagefile method directly)
explanation = lime:::lime.imagefile(image_path, model, image_preprocessor)

#explain image prediction
explained = lime:::explain.imagefile(image_path, explanation, 
                                     n_labels = 1, n_features = 5)
# Error in `*tmp*`[4, , ] : subscript out of bounds

