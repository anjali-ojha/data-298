GPU setup on aws cluster - 

types - https://aws.amazon.com/ec2/instance-types/


instructions - https://www.youtube.com/watch?v=gEdrvJTWHG4&ab_channel=datakulture

update system 

    sudo apt -y update

Install drivers 

    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-driver-types
    https://www.nvidia.com/en-us/drivers/

    sudo apt install nvidia-driver-550 nvidia-dkms-550

    restart the system and verify with nvidia-smi


Setup Python 

    source <env_name>/bin/activate


