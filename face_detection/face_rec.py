#!/usr/bin/python
# -*- coding:utf-8 -*-
import requests
import json


class FaceIdentify:
    def __init__(self):
        self._ak = "iF8TEFQQ9Lb9swBDgNGj4mZO"  # 百度云应用的AK
        self._sk = "K2YGG1R6GEZmDljvT3luiIXdPeW2snO2"  # 百度云应用的SK
        self._header = {'content-type': 'application/json'}
        self._search_url = "https://aip.baidubce.com/rest/2.0/face/v3/search"
        self._face_match_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
        self._face_add_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add"
        self._face_delete_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/face/delete"
        self._face_update_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/update"
        self._face_getlist_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/face/getlist"

    def get_token(self) -> str:
        """
        获取访问请求的token(access_token)
        :return: token内容
        """
        token_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="\
            + self._ak + "&client_secret=" + self._sk
        response = requests.get(token_url)
        access_token = eval(response.text)['access_token']
        return access_token

    def add_face(self, base64img, user_id, group_id, name):  # 注册人脸
        """
        注册人脸
        :param base64img: 图片的base64编码
        :param user_id: 添加用户的id
        :param group_id: 添加用户所在组的id
        :param name: 用户信息
        :return: 返回 人脸对应的token标识 以及 是否成功
        """
        access_token = self.get_token()
        request_url = self._face_add_url + "?access_token=" + access_token
        data = {'image': base64img,
                'image_type': 'BASE64',
                'group_id': str(group_id),
                'user_id': str(user_id),
                'user_info': str(name),  # 用户资料
                'quality_control': 'LOW',
                'liveness_control': 'NORMAL'}  # 留下基本格式
        response = requests.post(request_url, data=data, headers=self._header)
        response = response.text.replace("null", "\"null\"")
        flag = eval(response)['error_msg']
        if eval(response)["result"] != 'null':
            face_token = eval(response)["result"]['face_token']
        else:
            face_token = 'none'
        return face_token, flag  # flag是注册是否成功，face_token是每张人脸对应的基本标识

    def delete_face(self, face_token, user_id, group_id) -> bool:
        """
        删除人脸库中的人脸
        :param face_token: 人脸token标识
        :param user_id: 用户的id
        :param group_id: 所在组的id
        :return: 返回是否成功
        """
        data = {'user_id': str(user_id),
                'group_id': str(group_id),
                'face_token': str(face_token)}
        access_token = self.get_token()
        requests_url = self._face_delete_url + "?access_token=" + access_token
        response = requests.post(requests_url, data=data, headers=self._header).json()
        return eval(response)['error_msg']

    def get_face_token_list(self, user_id, group_id):
        """
        提供目标在人脸库的位置
        :param user_id: 用户id
        :param group_id: 所在组id
        :return: 返回对应face_token
        """
        data = {'user_id': str(user_id),
                'group_id': str(group_id)}
        access_token = self.get_token()
        requests_url = self._face_getlist_url + "?access_token=" + access_token
        response = requests.post(requests_url, data=data, headers=self._header).json()
        face_token_list = []
        if eval(response)["result"] != 'null':
            for i in eval(response)['result']['face_list']:
                face_token_list.append(i['face_token'])
        else:
            face_token_list = 'none'
        return face_token_list

    def compare_face(self, face1, face2) -> bool:
        """
        根据两张图片比对返回是否为同一个人
        如果连接失败会抛出 requests.RequestException
        :param face1: base64编码的人脸图
        :param face2: base64编码的人脸图
        :return: 是否为同一人
        """
        params = json.dumps([
            {"image": str(face1), "image_type": 'BASE64', "face_type": "LIVE"},
            {"image": str(face2), "image_type": 'BASE64', "face_type": "IDCARD"}
        ])
        access_token = self.get_token()
        requests_url = self._face_match_url + "?access_token=" + access_token
        response = requests.post(requests_url, data=params, headers=self._header).json()
        if response["error_msg"] == 'SUCCESS':
            score = response['result']['score']
            if score > 80:
                return True
            else:
                return False
        else:
            raise requests.RequestException("返回数据失败")

    def face_search(self, base64img, group_id):
        """
        通过图片在人脸库寻找对应的人脸
        如果连接失败会抛出 requests.RequestException
        :param base64img: 人脸图的base64编码
        :param group_id: 人脸搜索所在组
        :return: 人脸token标识
        """
        access_token = self.get_token()
        requests_url = self._search_url + "?access_token=" + access_token
        data = {'image': base64img,
                'image_type': 'BASE64',
                'group_id_list': str(group_id),
                'quality_control': 'LOW',
                'liveness_control': 'NORMAL'}
        response = requests.post(requests_url, data, headers=self._header).json()
        if response["error_msg"] == "SUCCESS" and response["result"] != "null":
            if response["result"]["user_list"][0]['score'] < 80:  # 符合分数以80为标准
                return 'none'  # 人脸库没有符合图片的人脸
            return response["result"]["user_list"][0]['user_id']
        else:
            raise requests.RequestException("返回数据失败")
