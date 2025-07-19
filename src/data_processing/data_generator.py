import json
import random
from typing import List, Dict, Any
from loguru import logger


class LogisticsDataGenerator:
    """物流填单数据生成器"""
    
    def __init__(self):
        self.names = [
            "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
            "郑一", "王二", "冯三", "陈四", "褚五", "卫六", "蒋七", "沈八",
            "韩九", "杨十", "朱一", "秦二", "尤三", "许四", "何五", "吕六",
            "施七", "张八", "孔九", "曹十", "严一", "华二", "金三", "魏四"
        ]
        
        self.cities = [
            "北京市", "上海市", "广州市", "深圳市", "杭州市", "南京市", "成都市", "武汉市",
            "西安市", "重庆市", "天津市", "苏州市", "无锡市", "宁波市", "青岛市", "大连市",
            "厦门市", "福州市", "济南市", "郑州市", "长沙市", "南昌市", "合肥市", "太原市",
            "石家庄市", "哈尔滨市", "长春市", "沈阳市", "呼和浩特市", "银川市", "西宁市", "兰州市"
        ]
        
        self.districts = [
            "朝阳区", "海淀区", "东城区", "西城区", "丰台区", "石景山区", "门头沟区", "房山区",
            "通州区", "顺义区", "昌平区", "大兴区", "怀柔区", "平谷区", "密云区", "延庆区",
            "黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区", "闵行区",
            "宝山区", "嘉定区", "浦东新区", "金山区", "松江区", "青浦区", "奉贤区", "崇明区"
        ]
        
        self.streets = [
            "中关村大街", "长安街", "王府井大街", "西单大街", "东单大街", "建国门外大街",
            "复兴门外大街", "西直门外大街", "东直门外大街", "朝阳门外大街", "阜成门外大街",
            "德胜门外大街", "安定门外大街", "永定门外大街", "广安门外大街", "右安门外大街",
            "左安门外大街", "南三环", "北三环", "东三环", "西三环", "南四环", "北四环",
            "东四环", "西四环", "南五环", "北五环", "东五环", "西五环"
        ]
        
        self.companies = [
            "科技有限公司", "贸易有限公司", "电子商务有限公司", "物流有限公司", "快递有限公司",
            "信息技术有限公司", "网络科技有限公司", "软件开发有限公司", "咨询服务有限公司",
            "投资管理有限公司", "金融服务有限公司", "教育培训有限公司", "医疗健康有限公司"
        ]
        
    def generate_name(self) -> str:
        """生成姓名"""
        return random.choice(self.names)
    
    def generate_phone(self) -> str:
        """生成手机号"""
        prefixes = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                   "150", "151", "152", "153", "155", "156", "157", "158", "159",
                   "180", "181", "182", "183", "184", "185", "186", "187", "188", "189"]
        prefix = random.choice(prefixes)
        suffix = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix}{suffix}"
    
    def generate_address(self) -> str:
        """生成地址"""
        city = random.choice(self.cities)
        district = random.choice(self.districts)
        street = random.choice(self.streets)
        number = random.randint(1, 999)
        return f"{city}{district}{street}{number}号"
    
    def generate_company(self) -> str:
        """生成公司名"""
        name = random.choice(self.names)
        company = random.choice(self.companies)
        return f"{name}{company}"
    
    def generate_logistics_form(self) -> Dict[str, Any]:
        """生成物流填单数据"""
        # 随机选择字段组合
        fields = []
        
        # 收件人信息
        if random.random() > 0.1:  # 90%概率包含收件人
            fields.append(f"收件人：{self.generate_name()}")
        
        if random.random() > 0.2:  # 80%概率包含收件人电话
            fields.append(f"收件人电话：{self.generate_phone()}")
        
        if random.random() > 0.1:  # 90%概率包含收件地址
            fields.append(f"收件地址：{self.generate_address()}")
        
        # 寄件人信息
        if random.random() > 0.3:  # 70%概率包含寄件人
            fields.append(f"寄件人：{self.generate_name()}")
        
        if random.random() > 0.4:  # 60%概率包含寄件人电话
            fields.append(f"寄件人电话：{self.generate_phone()}")
        
        if random.random() > 0.3:  # 70%概率包含寄件地址
            fields.append(f"寄件地址：{self.generate_address()}")
        
        # 公司信息
        if random.random() > 0.5:  # 50%概率包含公司
            fields.append(f"公司：{self.generate_company()}")
        
        # 其他信息
        if random.random() > 0.6:  # 40%概率包含备注
            remarks = ["易碎品", "轻拿轻放", "贵重物品", "生鲜", "急件", "到付"]
            fields.append(f"备注：{random.choice(remarks)}")
        
        if random.random() > 0.7:  # 30%概率包含重量
            weight = random.uniform(0.1, 20.0)
            fields.append(f"重量：{weight:.1f}kg")
        
        # 随机打乱字段顺序
        random.shuffle(fields)
        
        # 生成输入文本
        input_text = "，".join(fields)
        
        # 解析结构化信息
        structured_info = {}
        for field in fields:
            if "：" in field:
                key, value = field.split("：", 1)
                structured_info[key] = value
        
        return {
            "instruction": "请从以下物流填单信息中提取结构化数据",
            "input": input_text,
            "output": json.dumps(structured_info, ensure_ascii=False)
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成数据集"""
        logger.info(f"开始生成{num_samples}条训练数据...")
        dataset = []
        
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                logger.info(f"已生成 {i + 1} 条数据")
            dataset.append(self.generate_logistics_form())
        
        logger.info(f"数据生成完成，共生成 {len(dataset)} 条数据")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filepath: str):
        """保存数据集到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"数据集已保存到 {filepath}")


def main():
    """主函数，用于生成示例数据"""
    generator = LogisticsDataGenerator()
    
    # 生成训练数据
    train_data = generator.generate_dataset(5000)
    generator.save_dataset(train_data, "data/train.json")
    
    # 生成验证数据
    eval_data = generator.generate_dataset(500)
    generator.save_dataset(eval_data, "data/eval.json")
    
    # 生成测试数据
    test_data = generator.generate_dataset(500)
    generator.save_dataset(test_data, "data/test.json")
    
    # 打印示例
    print("示例数据：")
    for i in range(3):
        sample = generator.generate_logistics_form()
        print(f"样本 {i+1}:")
        print(f"  输入: {sample['input']}")
        print(f"  输出: {sample['output']}")
        print()


if __name__ == "__main__":
    main() 