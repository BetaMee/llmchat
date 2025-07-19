import json
import torch
import re
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger


class LogisticsExtractor:
    """物流信息提取器"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        logger.info(f"正在加载模型: {self.model_path}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def format_prompt(self, input_text: str) -> str:
        """格式化提示词"""
        instruction = "请从以下物流填单信息中提取结构化数据"
        prompt = f"<|im_start|>system\n你是一个专业的物流信息提取助手，能够从物流填单信息中准确提取结构化数据。<|im_end|>\n"
        prompt += f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt
    
    def extract_information(self, input_text: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        """提取物流信息"""
        try:
            # 格式化提示词
            prompt = self.format_prompt(input_text)
            
            # 分词
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码回复
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取assistant部分
            assistant_start = response.find("<|im_start|>assistant\n")
            if assistant_start != -1:
                assistant_text = response[assistant_start:].replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
            else:
                assistant_text = response
            
            # 解析JSON
            try:
                # 尝试直接解析
                result = json.loads(assistant_text)
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取JSON部分
                json_match = re.search(r'\{.*\}', assistant_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # 如果还是失败，返回原始文本
                        result = {"raw_response": assistant_text}
                else:
                    result = {"raw_response": assistant_text}
            
            return {
                "success": True,
                "extracted_info": result,
                "raw_response": assistant_text
            }
            
        except Exception as e:
            logger.error(f"信息提取失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_info": {},
                "raw_response": ""
            }
    
    def batch_extract(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        """批量提取信息"""
        results = []
        for text in input_texts:
            result = self.extract_information(text)
            results.append(result)
        return results
    
    def validate_extraction(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """验证提取结果"""
        validation_result = {
            "is_valid": True,
            "missing_fields": [],
            "invalid_fields": []
        }
        
        # 检查必要字段
        required_fields = ["收件人", "收件地址"]
        for field in required_fields:
            if field not in extracted_info or not extracted_info[field]:
                validation_result["missing_fields"].append(field)
                validation_result["is_valid"] = False
        
        # 验证电话号码格式
        phone_fields = ["收件人电话", "寄件人电话"]
        for field in phone_fields:
            if field in extracted_info and extracted_info[field]:
                phone = extracted_info[field]
                if not re.match(r'^1[3-9]\d{9}$', str(phone)):
                    validation_result["invalid_fields"].append(f"{field}: {phone}")
        
        return validation_result


def main():
    """主函数，用于测试推理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="物流信息提取推理")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--input_text", type=str, required=True,
                       help="输入文本")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备类型")
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = LogisticsExtractor(args.model_path, args.device)
    
    # 提取信息
    result = extractor.extract_information(args.input_text)
    
    # 打印结果
    print("输入文本:", args.input_text)
    print("提取结果:", json.dumps(result, ensure_ascii=False, indent=2))
    
    # 验证结果
    if result["success"]:
        validation = extractor.validate_extraction(result["extracted_info"])
        print("验证结果:", json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 