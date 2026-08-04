package com.dsapi.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class ServerService extends Service {
    private static final String TAG = "DeepSeekAPI";
    private static final String CHANNEL_ID = "dsapi_server";
    public static final int PORT = 8001;
    private static volatile boolean sStarted = false;

    public static boolean isStarted() { return sStarted; }

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
        startForeground(1, buildNotification("服务运行中 · 127.0.0.1:" + PORT));
        startPython();
    }

    private void startPython() {
        if (sStarted) return;
        new Thread(() -> {
            try {
                Context ctx = getApplicationContext();
                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(ctx));
                }
                Python py = Python.getInstance();
                PyObject boot = py.getModule("android_boot");

                String filesDir = ctx.getFilesDir().getAbsolutePath();
                // Node 可执行文件（libnodebin.so）位于 nativeLibraryDir，
                // 这是 Android 唯一允许执行二进制的目录
                String nativeLibDir = ctx.getApplicationInfo().nativeLibraryDir;
                Log.i(TAG, "nativeLibDir=" + nativeLibDir);

                sStarted = true;
                String r = boot.callAttr("start", filesDir, nativeLibDir, PORT).toString();
                Log.i(TAG, "python start -> " + r);
            } catch (Throwable t) {
                sStarted = false;
                Log.e(TAG, "python start failed", t);
            }
        }, "ds-boot").start();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID, "DeepSeek API 服务", NotificationManager.IMPORTANCE_LOW);
            ch.setShowBadge(false);
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) nm.createNotificationChannel(ch);
        }
    }

    private Notification buildNotification(String text) {
        Notification.Builder b = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return b.setContentTitle("DeepSeek API")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setOngoing(true)
                .build();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
