from fastai.vision.all import *

path = Path('chest_xray')

dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='val',
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(),
    num_workers=0
)

print("Classes found:", dls.vocab)

learn = vision_learner(dls, resnet34, metrics=accuracy)

learn.fine_tune(2)

learn.export('pneumonia_model.pkl')
print("Model saved!")