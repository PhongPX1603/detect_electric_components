# detect_electric_components

## Project Structure
```
    detect_electric_components
                    |
                    ├── config
                    |	  ├── electric_labelme_training.yaml      
                    |	  └── electric_labelme_testing.yaml
                    |
                    ├── flame
                    |	  ├── core 
                    |     |     ├── data 
                    |     |     |     ├── electric_components.py
                    |     |     |     └── visualize.py
                    |     |     |
                    |     |     ├── engine
                    |     |     |     ├── evaluator.py
                    |     |     |     └── trainer.py
                    |     |     |
                    |     |     ├── loss
                    |     |     |     ├── compute_iou.py
                    |     |     |     ├── generate_anchors.py
                    |     |     |     ├── loss.py
                    |     |     |     └── yolov3_loss.py
                    |     |     |
                    |     |     └── model
                    |     |           ├── darknet53.py
                    |     |           └── model.py
                    |     |     
                    |     |     
                    |	  └── handle               
                    |           ├── metrics
                    |           |     ├── loss
                    |           |     |     ├── loss.py
                    |           |     |     └── yolov3_loss.py
                    |           |     |
                    |           |     ├── mean_average_precision
                    |           |     |     ├── evaluator.py
                    |           |     |     └── mean_average_precision.py
                    |           |     |
                    |           |     └── metrics.py
                    |           |
                    |           ├── checkpoint.py
                    |           ├── early_stopping.py
                    |           ├── lr_scheduler.py
                    |           ├── metric_evaluator.py
                    |           ├── region_predictor.py
                    |           ├── screenlogger.py
                    |           └── terminate_on_nan.py
                    |   
                    |
                    ├── __main__.py
                    ├── module.py
                    └── utils.py
```
