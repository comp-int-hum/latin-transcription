import difflib
from termcolor import colored
import pytorch_lightning as L
import torch
import re
import numpy as np
from torchmetrics.text import CharErrorRate, WordErrorRate
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import glob
import os.path
import hashlib
from torch import optim, nn
from tqdm import tqdm

all_chars = " -.ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz¶"
char_to_num = {char: idx + 1 for idx, char in enumerate(all_chars)}
num_to_char = {idx + 1: char for idx, char in enumerate(all_chars)}

class LineImageDataset(Dataset):
    def classify(self, line_im_filename):
        str_hash = hashlib.md5(line_im_filename.encode()).hexdigest()
        hash_num = int(str_hash[:8], 16) % 100
        if hash_num < 90: return "train"
        else: return "val"

    def get_namespace(self, element):
        m = re.match('\{.*\}', element.tag)
        return m.group(0)[1:-1] if m else ''    

    def __init__(self, dirname, lines_dir, char_to_num, num_to_char, data_type, transform=None, return_filenames=False):
        self.transform = transform       
        self.char_to_num = char_to_num
        self.num_to_char = num_to_char
        self.data_type = data_type
        self.line_images = []
        self.line_image_filenames = []
        self.labels = []
        self.num_labels = []

        self.return_filenames = return_filenames
    
        #Iterate over all lines of all XML files
        for filename in tqdm(sorted(glob.glob(f"{dirname}/" + "*.xml"))):
            tree = ET.parse(filename)
            ns = {"ns": self.get_namespace(tree.getroot())}
            ET.register_namespace('', ns['ns'])
            root = tree.getroot()

            image_filename = root.find('ns:Page', ns).get('imageFilename')
 
            #First iteration: calculate average line spacing
            for text_region in root.findall('.//ns:TextRegion', ns):
                for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):      
                    new_image_filename = image_filename.split('.')[0]              
                    line_im_filename = "{}/{}_line_{}".format(lines_dir, new_image_filename, lineno)
                    line_im_filename, _ = os.path.splitext(line_im_filename)
                    line_im_filename += ".npy"
                    
                    if data_type != "all" and self.classify(line_im_filename) != data_type:
                        continue


                        
                    self.line_image_filenames.append(line_im_filename)  
                    self.line_images.append(torch.tensor(np.load(line_im_filename), dtype=torch.float32).unsqueeze(0))
                    text = text_line.find('.//ns:TextEquiv', ns).find('.//ns:Unicode', ns).text
                    text = text.strip()
                    text = text.replace(",", ".")
                    
                    
                    self.labels.append(text)
                    try:
                        self.num_labels.append(torch.tensor([self.char_to_num[c] for c in text]))
                    except:
                        print(text)
                        raise ValueError("Uh oh")

                                    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):                
        image = self.line_images[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.return_filenames:
            return {"image": image, "target": self.num_labels[idx], "text": self.labels[idx], "filename": self.line_image_filenames[idx]}
        return {"image": image, "target": self.num_labels[idx], "text": self.labels[idx]}


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (4,16), padding=(1,7)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 32, (4,16), padding=(1,7)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 64, (3,8), padding=(1,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(64, 64, (3,8), padding=(1,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
    
        self.lstms = nn.ModuleList([
            nn.LSTM(960, 256, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            nn.Dropout1d(0.3),
        ])
        self.lin = nn.Linear(512, len(all_chars) + 1)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(-1, x.shape[1] * x.shape[2], x.shape[3]).transpose(1,2)
        for layer in self.lstms:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.Dropout1d):
                x = x.transpose(1,2)
                assert(x.shape[1] == 512) #should be double LSTM hidden size
                x = layer(x)
                x = x.transpose(1,2)
            else:
                assert(False)
    
        x = self.lin(x)
        x = nn.functional.log_softmax(x, dim=2)
        return x.transpose(1,2)


class LatinTranscriber(L.LightningModule):
        def __init__(self, net, codec_l2c, ):
            super().__init__()
            self.codec_l2c = codec_l2c
            self.cer_calc = CharErrorRate()
            self.wer_calc = WordErrorRate()
            self.train_cer_calc = CharErrorRate()
            self.train_wer_calc = WordErrorRate()
            self.net = net
                
        def get_loss(self, batch, batch_idx):
            target = batch["target"]
            target_length = batch["target"].shape[1]
            input = batch["image"]
            output = self.net(input)
            output_length = output.shape[-1]
        
            loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)
            loss = loss_func(output.permute(2,0,1), target, (output_length,), (target_length,))
            return loss, output
    
        def on_train_epoch_start(self):
            self.train_cer_calc.reset()
            self.train_wer_calc.reset()
        
        
        def _get_current_lr(self):
            for param_group in self.trainer.optimizers[0].param_groups:
                return param_group['lr']
        
        def on_train_epoch_end(self):
            char_accuracy = max(1 - self.train_cer_calc.compute(), 0)
            word_accuracy = max(1 - self.train_wer_calc.compute(), 0)
            lr = self._get_current_lr()
        
            self.log("train_char_acc", char_accuracy, sync_dist=True)
            self.log("train_word_acc", word_accuracy, sync_dist=True)
            self.log('lr-Adam', lr, sync_dist=True)

 
        def training_step(self, batch, batch_idx):
            assert self.net.training
            loss, output = self.get_loss(batch, batch_idx)
            prediction, truth = self.get_prediction_and_truth(output, batch["target"])
            self.train_cer_calc.update(prediction, truth)
            self.train_wer_calc.update(prediction, truth)
            self.log("train_loss", loss, sync_dist=True)
            return loss
    
        def get_prediction_and_truth(self, output, target):
            target = torch.squeeze(target).cpu().numpy()      
            truth = ''.join([self.codec_l2c[target[i].item()] for i in range(len(target))])  
            labels = torch.argmax(torch.squeeze(output), axis=0).cpu().numpy()
            prediction = ""
            for i in range(len(labels)):
                label = labels[i]
                if label != 0 and (i==0 or label != labels[i-1]):
                    prediction += self.codec_l2c[label]
                
            return prediction, truth
    
        def validation_step(self, batch, batch_idx):
            assert not self.net.training
            assert batch["target"].shape[0] == 1
            loss, output = self.get_loss(batch, batch_idx)
            prediction, truth = self.get_prediction_and_truth(output, batch["target"])
            self.cer_calc.update(prediction, truth)
            self.wer_calc.update(prediction, truth)
            return loss
            
    
        def on_validation_epoch_start(self):
            self.cer_calc.reset()
            self.wer_calc.reset()
        
        def on_validation_epoch_end(self):
            char_accuracy = max(1 - self.cer_calc.compute(), 0) 
            word_accuracy = max(1 - self.wer_calc.compute(), 0)
            print("Epoch, char acc, word acc:", self.current_epoch, round(char_accuracy.item(), 4), round(word_accuracy.item(), 4))
            self.log("val_char_acc", char_accuracy, sync_dist=True)
            self.log("val_word_acc", word_accuracy, sync_dist=True)


        def configure_optimizers(self):          
            optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                "name": "lr-scheduler",
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.34, min_lr=1e-4, patience=15),
                "monitor": "val_word_acc",
                "frequency": 1
                },
            }




def highlight_differences(pred, truth, level='char'):
    if level == 'char':
        seqm = difflib.SequenceMatcher(None, truth, pred)
        output = []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(truth[a0:a1])
            elif opcode == 'insert':
                inserted_text = pred[b0:b1]
                output.append(colored(inserted_text, 'green', attrs=['bold', 'underline']))
            elif opcode == 'delete':
                deleted_text = truth[a0:a1]
                output.append(colored(deleted_text, 'red', attrs=['bold', 'underline']))
            elif opcode == 'replace':
                replaced_text = pred[b0:b1]
                output.append(colored(replaced_text, 'yellow', attrs=['bold', 'underline']))
        return ''.join(output)
    elif level == 'word':
        truth_words = truth.split()
        pred_words = pred.split()
        seqm = difflib.SequenceMatcher(None, truth_words, pred_words)
        output = []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.extend(truth_words[a0:a1])
            elif opcode == 'insert':
                inserted_words = ' '.join(pred_words[b0:b1])
                output.append(colored(inserted_words, 'green', attrs=['bold', 'underline']))
            elif opcode == 'delete':
                deleted_words = ' '.join(truth_words[a0:a1])
                output.append(colored(deleted_words, 'red', attrs=['bold', 'underline']))
            elif opcode == 'replace':
                replaced_words = ' '.join(pred_words[b0:b1])
                output.append(colored(replaced_words, 'yellow', attrs=['bold', 'underline']))
        return ' '.join(output)
    else:
        raise ValueError("Invalid level: choose 'char' or 'word'")