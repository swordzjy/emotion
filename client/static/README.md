# 静态设计页面 - Video Scene App

基于 `main.pen` 设计稿导出的静态 HTML 页面，支持移动端响应式布局。已集成 orangutan 项目的办公场景视频播放功能。

## 结构

```
static/
├── index.html      # 主页面（含场景入口）
├── scene.html      # 办公场景视频播放（源自 orangutan）
├── video/          # 视频资源（video1.mp4 ~ video6.mp4）
├── css/
│   └── app.css     # 样式（含移动端适配）
├── js/
│   └── app.js      # 基础交互（图标、标签切换、手风琴）
└── README.md
```

## 办公场景入口

点击首页「场景与入口」中的 **办公室场景** 卡片，即可进入视频播放页面。在播放页面中：

- **点击人物部位**：头部、胸部、左臂、右臂、腰部以下，触发对应剧情反应
- **键盘 1-5**：对应上述五个部位

## 兼容性

- **iPhone**：支持刘海屏安全区域（safe-area-inset）
- **Android**：主流机型
- **视口**：`viewport-fit=cover` + `width=device-width` + `initial-scale=1`
- **触摸**：触摸目标 ≥ 44×44px

## 使用方式

### 直接打开
用浏览器打开 `index.html` 即可（需联网加载字体与图标）。

### 本地服务
```bash
cd client/static
python -m http.server 8080
# 访问 http://localhost:8080
```

### 在 emotion 项目中
若已有服务器，将 `static` 作为静态资源目录部署即可。
