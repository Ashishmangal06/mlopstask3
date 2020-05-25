FROM centos:latest

RUN yum install python36 -y

RUN yum install -y epel-release

RUN yum groupinstall -y 'development tools'

RUN yum install -y python36-devel

RUN yum install python3-pip -y

RUN yum install python3-setuptools -y

RUN yum install python3-wheel -y

RUN yum install pkg-config -y

RUN yum install sudo -y 

RUN pip3 install keras

RUN pip3 install numpy

RUN pip3 install scipy

RUN pip3 install sklearn

RUN pip3 install pandas

RUN pip3 install scikit-learn

RUN pip3 install pillow

RUN pip3 install opencv-python

RUN pip3 install tensorflow==1.12.0
RUN pip3 install --upgrade tensorflow-probablity

RUN pip3 install matplotlib

COPY . /dataset

CMD ["python3", "/mlops/ipynb.py"]
