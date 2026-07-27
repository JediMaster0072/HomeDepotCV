from prometheus_client import Counter, Histogram

# [Count metrics]
total_images_count = Counter('cv_singleline_total_images_count', "Total images received from bucket",
                             ["experience", "sub_experience", "application", "environment"])
total_ack_message_count = Counter('cv_singleline_total_ack_count', "Total acknowledged message count",
                                  ["experience", "sub_experience", "application", "environment"])
total_nack_message_count = Counter('cv_singleline_total_nack_count', "Total negative acknowledged message count",
                                   ["experience", "sub_experience", "application", "environment"])
total_error_count = Counter('cv_singleline_total_error_count', "Total error count",
                            ["experience", "sub_experience", "application", "environment", "errorInfo"])
total_acceptable_count = Counter('cv_singleline_acceptable_count', "Acceptable count",
                                 ["experience", "sub_experience", "application", "environment", "vol", "clip_iqa",
                                  "acceptable"])

# [Processing time metrics]
overall_process_time = Histogram('cv_singleline_overall_process_time', 'Over all process time',
                                 ["experience", "sub_experience", "application", "environment"])
download_image_time = Histogram('cv_singleline_download_image_time', 'Download image time',
                                ["experience", "sub_experience", "application", "environment"])
get_clip_iqa_time = Histogram('cv_singleline_clip_iqa_time', 'Get clip iqa score time',
                              ["experience", "sub_experience", "application", "environment"])
get_vol_time = Histogram('cv_singleline_vol_time', 'Get vol score time',
                         ["experience", "sub_experience", "application", "environment"])
