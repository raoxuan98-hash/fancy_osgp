"""
测试运行器，用于解决导入问题
"""

import sys
import os
import unittest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_test_module(module_name, test_classes=None):
    """运行指定的测试模块"""
    try:
        # 导入测试模块
        test_module = __import__(module_name, fromlist=[''])
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        if test_classes:
            # 运行指定的测试类
            for test_class in test_classes:
                suite.addTests(loader.loadTestsFromTestCase(test_class))
        else:
            # 运行模块中的所有测试
            suite = loader.loadTestsFromModule(test_module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful(), result
    except Exception as e:
        print(f"运行测试模块 {module_name} 时出错: {e}")
        return False, None

def run_all_tests():
    """运行所有测试"""
    test_modules = [
        {
            "name": "数据集类型自动检测测试",
            "module": "tests.reference_dataset.test_dataset_detection",
            "classes": ["TestDatasetTypeDetector", "TestPathValidator"]
        },
        {
            "name": "数据集工厂测试",
            "module": "tests.reference_dataset.test_dataset_factory",
            "classes": ["TestReferenceDatasetFactory"]
        },
        {
            "name": "配置参数解析测试",
            "module": "tests.reference_dataset.test_config_parsing",
            "classes": ["TestConfigParsing"]
        },
        {
            "name": "错误处理测试",
            "module": "tests.reference_dataset.test_error_handling",
            "classes": ["TestErrorHandling"]
        },
        {
            "name": "向后兼容性测试",
            "module": "tests.reference_dataset.test_backward_compatibility",
            "classes": ["TestBackwardCompatibility"]
        }
    ]
    
    all_results = []
    
    for test_info in test_modules:
        print("\n" + "=" * 60)
        print(f"运行 {test_info['name']}")
        print("=" * 60)
        
        try:
            # 动态导入测试类
            test_module = __import__(test_info['module'], fromlist=[''])
            test_classes = []
            for class_name in test_info['classes']:
                test_classes.append(getattr(test_module, class_name))
            
            success, result = run_test_module(test_info['module'], test_classes)
            all_results.append((test_info['name'], success, result))
            
            if success:
                print(f"\n✅ {test_info['name']} 通过！")
            else:
                print(f"\n❌ {test_info['name']} 失败")
                if result:
                    print(f"   失败: {len(result.failures)} 个, 错误: {len(result.errors)} 个")
                    
        except Exception as e:
            print(f"\n❌ {test_info['name']} 运行出错: {e}")
            all_results.append((test_info['name'], False, None))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, success, _ in all_results if success)
    
    for test_name, success, _ in all_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试模块通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试模块都通过了！")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试模块失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)