module.exports = {
  apps: [{
    name: 'emotion-api',
    script: '/home/ubuntu/Emotion/emotion/.myenv/bin/python',
    args: '-m uvicorn server.app:app --host 0.0.0.0 --port 8000',
    cwd: '/home/ubuntu/Emotion/emotion',  // 关键：设置工作目录
    interpreter: 'none',
    env: {
      PYTHONPATH: '/home/ubuntu/Emotion/emotion',  // 关键：设置Python路径
      PYTHONUNBUFFERED: '1'
    },
    error_file: '/home/ubuntu/Emotion/emotion/logs/error.log',
    out_file: '/home/ubuntu/Emotion/emotion/logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    autorestart: true,
    watch: false,
    max_memory_restart: '500M'
  }]
}