# Pipeline order: 03.1 and 46
# Description: Prints volume, duration, and elapsed-time metrics in the service's expected log format.
def log_metric(argument0, argument1, argument2):
    if argument2 is None:
        metric_text = str(argument1).upper()
        metric_value = argument0
        print(f'METRIC/{metric_text}/{metric_value}')
    else:
        metric_text = str(argument2).upper()
        metric_value = argument1 - argument0
        print(f'METRIC/{metric_text}/{metric_value}')