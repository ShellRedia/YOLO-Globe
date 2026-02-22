from ultralytics import YOLO
from peft import LoraConfig, get_peft_model
from torchinfo import summary
from ultralytics.nn.tasks import DetectionModel

class YOLO_Trainer:
    def __init__(self, weight_path="checkpoints/yolo26m.pt", device_ids=[0], dataset_name="globe_real"):
        self.yolo = YOLO(weight_path)  # load a pretrained model (recommended for training)
        self.data_path = "datasets/{}/{}.yaml".format(dataset_name, dataset_name)
        self.device_ids = device_ids
    
    def fine_tune(self):
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["conv"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
        self.print_trainable_parameters(self.yolo.model)
        self.yolo.model = get_peft_model(self.yolo.model, lora_config)
        self.print_trainable_parameters(self.yolo.model)

        DetectionModel.load = lambda self, weights: None
        self.train()


    def train(self):
        results = self.yolo.train(data=self.data_path, epochs=100, device=self.device_ids, imgsz=[540, 960], batch=1, save_period=10, plots=True, patience=10000)
    
    def resume(self):
        results = self.yolo.train(resume=True, device=self.device_ids)
    
    def print_trainable_parameters(self, model):
        # summary(model, (1, 3, 640, 480))
        trainable_params, all_param = 0, 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


if __name__=="__main__":
    yolo_trainer = YOLO_Trainer()
    yolo_trainer = YOLO_Trainer(weight_path="runs/detect/train/weights/best.pt")
    

    # yolo_trainer.train()
    yolo_trainer.fine_tune()