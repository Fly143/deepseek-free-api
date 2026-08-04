package com.dsapi.app;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final String BASE = "http://127.0.0.1:" + ServerService.PORT;
    private static final String ADMIN_USER = "admin";
    private static final String ADMIN_PASS = "admin";
    private static final int REQ_LOGIN = 1001;

    private WebView web;
    private TextView status;
    private Button loginBtn;
    private final Handler ui = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        web = findViewById(R.id.web);
        status = findViewById(R.id.status);
        loginBtn = findViewById(R.id.btn_login);
        loginBtn.setOnClickListener(v ->
                startActivityForResult(new Intent(this, LoginActivity.class), REQ_LOGIN));

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);

        // 不设 WebChromeClient 时 JS 的 confirm() 恒返回 false，
        // 管理页"删除账号"等带确认框的操作会静默失效
        web.setWebChromeClient(new android.webkit.WebChromeClient() {
            @Override
            public boolean onJsAlert(WebView v, String url, String msg,
                                     android.webkit.JsResult result) {
                new androidx.appcompat.app.AlertDialog.Builder(MainActivity.this)
                        .setMessage(msg)
                        .setPositiveButton("确定", (d, w) -> result.confirm())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }

            @Override
            public boolean onJsConfirm(WebView v, String url, String msg,
                                       android.webkit.JsResult result) {
                new androidx.appcompat.app.AlertDialog.Builder(MainActivity.this)
                        .setMessage(msg)
                        .setPositiveButton("确定", (d, w) -> result.confirm())
                        .setNegativeButton("取消", (d, w) -> result.cancel())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }

            @Override
            public boolean onJsPrompt(WebView v, String url, String msg, String defaultValue,
                                      android.webkit.JsPromptResult result) {
                final android.widget.EditText et = new android.widget.EditText(MainActivity.this);
                et.setText(defaultValue);
                new androidx.appcompat.app.AlertDialog.Builder(MainActivity.this)
                        .setMessage(msg)
                        .setView(et)
                        .setPositiveButton("确定", (d, w) -> result.confirm(et.getText().toString()))
                        .setNegativeButton("取消", (d, w) -> result.cancel())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }
        });

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
                return false;
            }

            @Override
            public void onReceivedHttpAuthRequest(WebView v,
                                                  android.webkit.HttpAuthHandler handler,
                                                  String host, String realm) {
                handler.proceed(ADMIN_USER, ADMIN_PASS);
            }
        });

        askNotificationPermission();
        startService(new Intent(this, ServerService.class));
        waitForServer();
    }

    private void askNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 &&
                checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.POST_NOTIFICATIONS}, 2001);
        }
    }

    /** 后台轮询服务端口，就绪后决定进登录页还是管理页。 */
    private void waitForServer() {
        status.setText("正在启动 DeepSeek API 服务…\n首次启动需解压运行时，请稍候");
        new Thread(() -> {
            boolean ok = false;
            for (int i = 0; i < 180; i++) {
                if (ping()) { ok = true; break; }
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                final int sec = i + 1;
                if (sec % 5 == 0) {
                    ui.post(() -> status.setText(
                            "正在启动 DeepSeek API 服务…（" + sec + "s）\n首次启动需解压 Node 运行时"));
                }
            }
            final boolean started = ok;
            ui.post(() -> {
                if (!started) {
                    status.setText("服务启动超时\n请杀掉应用重试，或查看日志");
                    return;
                }
                routeByAccounts();
            });
        }, "ds-wait").start();
    }

    /** 有账号 -> 直接进管理页；无账号 -> 进内置浏览器登录。 */
    private void routeByAccounts() {
        // 无论有无账号都直接进管理后台；
        // 登录入口在右下角"登录 DeepSeek"按钮，不再自动跳转登录页
        ui.post(this::showAdmin);
    }

    private void showAdmin() {
        status.setVisibility(View.GONE);
        web.setVisibility(View.VISIBLE);
        loginBtn.setVisibility(View.VISIBLE);
        loadAdmin();
    }

    /** 带 Basic 认证头加载管理页（WebView 不会自动带凭据）。 */
    private void loadAdmin() {
        String cred = ADMIN_USER + ":" + ADMIN_PASS;
        String b64 = Base64.encodeToString(cred.getBytes(), Base64.NO_WRAP);
        Map<String, String> h = new HashMap<>();
        h.put("Authorization", "Basic " + b64);
        web.loadUrl(BASE + "/admin", h);
    }

    private boolean ping() {
        HttpURLConnection c = null;
        try {
            c = (HttpURLConnection) new URL(BASE + "/admin").openConnection();
            c.setConnectTimeout(1500);
            c.setReadTimeout(1500);
            c.setRequestMethod("GET");
            int code = c.getResponseCode();
            // 401 也代表服务已经起来了
            return code > 0;
        } catch (Exception e) {
            return false;
        } finally {
            if (c != null) c.disconnect();
        }
    }

    private boolean hasAccount() {
        HttpURLConnection c = null;
        try {
            c = (HttpURLConnection) new URL(BASE + "/api/accounts").openConnection();
            c.setConnectTimeout(4000);
            c.setReadTimeout(6000);
            String cred = ADMIN_USER + ":" + ADMIN_PASS;
            c.setRequestProperty("Authorization",
                    "Basic " + Base64.encodeToString(cred.getBytes(), Base64.NO_WRAP));
            if (c.getResponseCode() != 200) return false;
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) sb.append(line);
            }
            String body = sb.toString();
            // 有账号时返回的数组里会带 account_label 字段
            return body.contains("account_label");
        } catch (Exception e) {
            return false;
        } finally {
            if (c != null) c.disconnect();
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == REQ_LOGIN) {
            showAdmin();
        }
    }

    @Override
    public void onRequestPermissionsResult(int rc, @NonNull String[] p, @NonNull int[] g) {
        super.onRequestPermissionsResult(rc, p, g);
    }

    @Override
    public void onBackPressed() {
        if (web.getVisibility() == View.VISIBLE && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
