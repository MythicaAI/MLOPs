from PIL import Image 
import requests 
from io import BytesIO
import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict

import labelbox as lb
import labelbox.types as lb_types
from labelbox.data.serialization.labelbox_v1.converter import LBV1Converter
from labelbox.data.serialization import COCOConverter

import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from coco_utils import separate_coco_semantic_from_panoptic

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

    

def cache(labelbox_api_key,
          labelbox_project_id,
          labelbox_cache_path
          ):
    lb_client = lb.Client(api_key=labelbox_api_key)
    project = lb_client.get_project(labelbox_project_id)
    project_export = project.export_labels(download=True)
    image_root = os.path.join(labelbox_cache_path,'images')
    mask_root = os.path.join(labelbox_cache_path,'masks')
    segmentation_root = os.path.join(labelbox_cache_path,'segmentation')


    def posix_path_serializer(obj):
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


    labels = LBV1Converter.deserialize(project_export)
        
    panoptic_labels = []
    for idx, label in enumerate(labels):
        panoptic_labels.append(
            lb_types.Label(
                data = label.data,
                annotations = [annot for annot in label.annotations if isinstance(annot.value, lb_types.Mask)]
                )
        )  

    coco_panoptic = COCOConverter.serialize_panoptic(
        labels = panoptic_labels,
        image_root = image_root,
        mask_root = mask_root,
        all_stuff = True,
        ignore_existing_data = True
    )

    train_json_panoptic_path=os.path.join(labelbox_cache_path,'json_train_panoptic.json')

    with open(train_json_panoptic_path, 'w') as file:
        json.dump(coco_panoptic, file,default=posix_path_serializer)

    separate_coco_semantic_from_panoptic(train_json_panoptic_path, mask_root, segmentation_root, coco_panoptic['categories'])


def train(source_model_name, 
          target_model_name,
          model_cache_path, 
          labelbox_cache_path
          ):

    class LabelBoxDataset(Dataset):
        def __init__(self, cache_path, processor):
            self.processor = processor
            self.images = [os.path.join(cache_path,'images',name) for name in os.listdir(os.path.join(cache_path,'images'))]
            self.masks = [os.path.join(cache_path,'masks',name) for name in os.listdir(os.path.join(cache_path,'masks'))]
            self.segmentation = [os.path.join(cache_path,'segmentation',name) for name in os.listdir(os.path.join(cache_path,'segmentation'))]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):

            img_url = self.images[idx]
            img = load_image(img_url)
            img = np.array(img)
            seg_url = self.segmentation[idx]
            seg = load_image(seg_url)
            seg = np.array(seg)
            #inputs = self.processor(images=[image], segmentation_maps=[label_mask], task_inputs=["semantic"], return_tensors="pt")
            inputs = self.processor(images=img, segmentation_maps=seg, task_inputs=["panoptic"] , return_tensors="pt")
            inputs = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

            #Pad class_labels
            desired_length = 18
            current_length = inputs['class_labels'].shape[0]
            padding_needed = max(desired_length - current_length, 0)
            if padding_needed > 0:
                # inputs['class_labels'] is a 1D tensor and needs padding at the end to reach the desired length
                inputs['class_labels'] = F.pad(inputs['class_labels'], (0, padding_needed), "constant", 0)

            # Calculate the current number of classes (x) and the padding needed
            current_length = inputs['mask_labels'].shape[0]
            padding_needed = max(desired_length - current_length, 0)
            if padding_needed > 0:
                # Pad only the class dimension (first dimension)
                # The pad sequence is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # Since we're padding the first dimension (class/channel dimension in this case), 
                # we use pad_front and pad_back, and set the rest to 0.
                padded_mask = F.pad(inputs['mask_labels'], (0, 0, 0, 0, 0, padding_needed), "constant", 0)
                inputs['mask_labels'] = padded_mask

            return inputs

    processor = AutoProcessor.from_pretrained(source_model_name, cache_dir= model_cache_path) 
    model = AutoModelForUniversalSegmentation.from_pretrained(source_model_name, cache_dir= model_cache_path, is_training=True)
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx 
    dataset = LabelBoxDataset(cache_path=labelbox_cache_path, processor=processor) 
    dataloader = DataLoader(dataset, batch_size=10)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    model.to(device)
    for epoch in range(10):  # loop over the dataset multiple times
        for batch in dataloader:

            # zero the parameter gradients
            optimizer.zero_grad()
            batch = {k:v.to(device) for k,v in batch.items()}

            # forward pass
            outputs = model(**batch)

            # backward pass + optimize
            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
    
    processor.save_pretrained(save_directory= os.path.join(model_cache_path,target_model_name))
    model.save_pretrained(save_directory= os.path.join(model_cache_path,target_model_name))


        
def infer(model_name, 
          model_cache_path,
          image_url,
          output_dir=None
          ):


    class Visualizer:
        @staticmethod
        def extract_legend(handles):
            fig = plt.figure()
            fig.legend(handles=handles, ncol=len(handles) // 20 + 1, loc='center')
            fig.tight_layout()
            return fig
        
        @staticmethod
        def predicted_semantic_map_to_figure(predicted_map):
            segmentation = predicted_map[0]
            # get the used color map
            viridis = plt.get_cmap('viridis', max(1, torch.max(segmentation)))
            # get all the unique numbers
            labels_ids = torch.unique(segmentation).tolist()
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            handles = []
            for label_id in labels_ids:
                label = id2label[label_id]
                color = viridis(label_id)
                handles.append(mpatches.Patch(color=color, label=label))
            fig_legend = Visualizer.extract_legend(handles=handles)
            fig.tight_layout()
            return fig, fig_legend
            
        @staticmethod
        def predicted_instance_map_to_figure(predicted_map):
            segmentation = predicted_map[0]['segmentation']
            segments_info = predicted_map[0]['segments_info']
            # get the used color map
            viridis = plt.get_cmap('viridis', max(torch.max(segmentation), 1))
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            instances_counter = defaultdict(int)
            handles = []
            # for each segment, draw its legend
            for segment in segments_info:
                segment_id = segment['id']
                segment_label_id = segment['label_id']
                segment_label = id2label[segment_label_id]
                label = f"{segment_label}-{instances_counter[segment_label_id]}"
                instances_counter[segment_label_id] += 1
                color = viridis(segment_id)
                handles.append(mpatches.Patch(color=color, label=label))
                
            fig_legend = Visualizer.extract_legend(handles)
            fig.tight_layout()
            return fig, fig_legend

        @staticmethod
        def predicted_panoptic_map_to_figure(predicted_map):
            segmentation = predicted_map[0]['segmentation']
            segments_info = predicted_map[0]['segments_info']
            # get the used color map
            viridis = plt.get_cmap('viridis', max(torch.max(segmentation), 1))
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            instances_counter = defaultdict(int)
            handles = []
            # for each segment, draw its legend
            for segment in segments_info:
                segment_id = segment['id']
                segment_label_id = segment['label_id']
                segment_label = id2label[segment_label_id]
                label = f"{segment_label}-{instances_counter[segment_label_id]}"
                instances_counter[segment_label_id] += 1
                color = viridis(segment_id)
                handles.append(mpatches.Patch(color=color, label=label))
                
            fig_legend = Visualizer.extract_legend(handles)
            fig.tight_layout()
            return fig, fig_legend

        @staticmethod
        def figures_to_images(fig, fig_legend, name_suffix=""):
            seg_filepath = os.path.join(output_dir,f"{name_suffix}_segmentation.png")
            leg_filepath = os.path.join(output_dir,f"{name_suffix}_legend.png")
            fig.savefig(seg_filepath)
            fig_legend.savefig(leg_filepath)
            segmentation = Image.open(seg_filepath)
            legend = Image.open(leg_filepath)
            return segmentation, legend


    processor = AutoProcessor.from_pretrained(model_name, cache_dir= model_cache_path) 
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name, cache_dir= model_cache_path, is_training=False)
    model.eval()
    id2label = model.config.id2label
    
    os.makedirs(output_dir, exist_ok=True)
    
    # set is_training attribute of base OneFormerModel to None after training
    # this disables the text encoder and hence enables to do forward passes
    # without passing text_inputs
    model.model.is_training = False

    # load image
    src_image = load_image(image_url)
    src_image_filename = 'oneformer_job'

    try:
        src_image_filename, _ = os.path.splitext(os.path.basename(image_url))

    except Exception as e:
        print(f"could not interpret filename. This is ok.")        

    # prepare image for the model
    inputs = processor(images=src_image, task_inputs=["panoptic"], return_tensors="pt")

    for k,v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(k,v.shape)

    # forward pass (no need for gradients at inference time)
    with torch.no_grad():
        outputs = model(**inputs)
  
    # postprocessing
    segmentation = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[src_image.size[::-1]])

    
    fig, fig_leg = Visualizer.predicted_panoptic_map_to_figure(segmentation)
    image,legend = Visualizer.figures_to_images(fig, fig_leg,name_suffix=src_image_filename)
    
    return image,legend



def load_image(url):
    """Load an image from a URL or a local file path."""
    if url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    elif url.startswith('file://'):
        file_path = url.replace('file://', '')
        image = Image.open(file_path)
    else:
        # Assuming it's a direct file path without 'file://' scheme
        image = Image.open(url)

    return image
