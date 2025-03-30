import albumentations as albu

class AbstractTransformWrapper:
    def __init__(self, param_list):
        self.param_list = param_list
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __str__(self):
        return self.__class__.__name__[:-7] # 返回类名
    
    def __next__(self):
        if self.index < len(self.param_list):
            param = self.param_list[self.index]
            self.index += 1
            
            if param == 0:
                return param, None
            else:
                return self._get_transform(param)
        else:
            self.index = 0
            raise StopIteration
    
    def _get_transform(self, param):
        raise NotImplementedError
    
        
class GaussianBlurWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussianBlur(
            blur_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class GaussianNoiseWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussNoise(
            var_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class JpegCompressionWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.JpegCompression(
            quality_lower = param-1,
            quality_upper = param,
            p=1.0
        )
    
if __name__ == "__main__":
    # 示例用法
    param_list = [90, 80, 0]  # 传入一个装有int的列表，代表需要遍历的参数数值
    wrapper = JpegCompressionWrapper(param_list)

    for transform in wrapper:
        print(transform)


    for transform in wrapper:
        print(transform)
        
    print(str(wrapper))
        
