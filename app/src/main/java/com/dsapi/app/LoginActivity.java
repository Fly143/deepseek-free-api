package com.dsapi.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.webkit.CookieManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/** 内置浏览器：登录 chat.deepseek.com 后自动抓取 token 并导入。 */
public class LoginActivity extends AppCompatActivity {

    private static final String TAG = "DeepSeekAPI";
    private static final String DS_URL = "https://chat.deepseek.com/sign_in";
    private static final String BASE = "http://127.0.0.1:" + ServerService.PORT;
    private static final String MOBILE_UA =
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                    + "(KHTML, like Gecko) Chrome/134.0.0.0 Mobile Safari/537.36";

    private WebView web;
    private TextView status;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private volatile boolean importing = false;
    private volatile boolean done = false;
    private Runnable poller;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        web = findViewById(R.id.web);
        status = findViewById(R.id.status);
        findViewById(R.id.btn_reload).setOnClickListener(v -> web.reload());
        findViewById(R.id.btn_clear).setOnClickListener(v -> {
            CookieManager cm = CookieManager.getInstance();
            cm.removeAllCookies(null);
            cm.flush();
            web.clearCache(true);
            web.evaluateJavascript("try{localStorage.clear();sessionStorage.clear()}catch(e){}", null);
            web.loadUrl(DS_URL);
            status.setText("已清除登录状态，请重新登录");
        });

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setUserAgentString(MOBILE_UA);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);

        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(web, true);

        web.setWebChromeClient(new android.webkit.WebChromeClient() {
            @Override
            public boolean onJsAlert(WebView v, String url, String msg,
                                     android.webkit.JsResult result) {
                new androidx.appcompat.app.AlertDialog.Builder(LoginActivity.this)
                        .setMessage(msg)
                        .setPositiveButton("确定", (d, w) -> result.confirm())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }

            @Override
            public boolean onJsConfirm(WebView v, String url, String msg,
                                       android.webkit.JsResult result) {
                new androidx.appcompat.app.AlertDialog.Builder(LoginActivity.this)
                        .setMessage(msg)
                        .setPositiveButton("确定", (d, w) -> result.confirm())
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
            public void onPageFinished(WebView v, String url) {
                tryGrab();
            }
        });

        status.setText("请在下方登录 DeepSeek 账号\n登录成功后会自动导入");
        web.loadUrl(DS_URL);

        // 登录多为页面内 JS 跳转，不触发 onPageFinished，故用轮询兜底
        poller = new Runnable() {
            @Override
            public void run() {
                if (done) return;
                tryGrab();
                ui.postDelayed(this, 1500);
            }
        };
        ui.postDelayed(poller, 2000);
    }

    /** 读取 localStorage 里的 userToken。 */
    private void tryGrab() {
        if (importing || done) return;
        web.evaluateJavascript(
                "(function(){try{"
                        + "var keys=['userToken','user_token','token'];"
                        + "for(var i=0;i<keys.length;i++){"
                        + "  var v=localStorage.getItem(keys[i]);"
                        + "  if(v){return v;}"
                        + "}"
                        // DeepSeek 有时把 token 包在 JSON 对象里
                        + "for(var j=0;j<localStorage.length;j++){"
                        + "  var k=localStorage.key(j);"
                        + "  var s=localStorage.getItem(k);"
                        + "  if(s&&s.indexOf('token')>=0&&s.length<4000){return '@@'+k+'@@'+s;}"
                        + "}"
                        + "return '';}catch(e){return '';}})()",
                value -> onJsValue(value));
    }

    private void onJsValue(String raw) {
        if (raw == null || raw.length() < 4) return;
        // evaluateJavascript 返回的是 JSON 字符串字面量
        String v = raw;
        if (v.startsWith("\"") && v.endsWith("\"")) {
            v = v.substring(1, v.length() - 1);
        }
        v = v.replace("\\\"", "\"").replace("\\\\", "\\").replace("\\n", "");
        if (v.isEmpty() || "null".equals(v)) return;

        String token = extractToken(v);
        if (token == null || token.length() < 20) return;

        importing = true;
        ui.post(() -> status.setText("检测到登录态，正在导入…"));
        final String tk = token;
        new Thread(() -> doImport(tk), "ds-import").start();
    }

    /** 从原始值中解析出 token 字符串。 */
    private String extractToken(String v) {
        try {
            if (v.startsWith("@@")) {
                int e = v.indexOf("@@", 2);
                String body = e > 0 ? v.substring(e + 2) : v;
                return pickFromJson(body);
            }
            if (v.trim().startsWith("{")) {
                return pickFromJson(v);
            }
            return v.trim();
        } catch (Exception e) {
            return null;
        }
    }

    private String pickFromJson(String body) {
        try {
            JSONObject o = new JSONObject(body);
            String[] cands = {"userToken", "token", "value", "access_token"};
            for (String c : cands) {
                if (o.has(c)) {
                    Object x = o.get(c);
                    if (x instanceof String && ((String) x).length() > 20) return (String) x;
                    if (x instanceof JSONObject) {
                        JSONObject oo = (JSONObject) x;
                        for (String c2 : cands) {
                            if (oo.has(c2) && oo.getString(c2).length() > 20) {
                                return oo.getString(c2);
                            }
                        }
                    }
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    private void doImport(String token) {
        HttpURLConnection c = null;
        try {
            c = (HttpURLConnection) new URL(BASE + "/api/accounts/import-token").openConnection();
            c.setRequestMethod("POST");
            c.setConnectTimeout(8000);
            c.setReadTimeout(40000);
            c.setDoOutput(true);
            c.setRequestProperty("Content-Type", "application/json");
            String cred = "admin:admin";
            c.setRequestProperty("Authorization",
                    "Basic " + Base64.encodeToString(cred.getBytes(), Base64.NO_WRAP));

            JSONObject body = new JSONObject();
            body.put("token", token);
            body.put("user_agent", MOBILE_UA);
            try (OutputStream os = c.getOutputStream()) {
                os.write(body.toString().getBytes("UTF-8"));
            }

            int code = c.getResponseCode();
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(
                    code == 200 ? c.getInputStream() : c.getErrorStream()))) {
                String line;
                while ((line = r.readLine()) != null) sb.append(line);
            }
            final String resp = sb.toString();
            Log.i(TAG, "import-token -> " + code + " " + resp);

            if (code == 200) {
                done = true;
                ui.post(() -> {
                    status.setText("✅ 导入成功，正在进入管理页…");
                    setResult(Activity.RESULT_OK);
                    ui.postDelayed(this::finish, 800);
                });
            } else {
                importing = false;
                ui.post(() -> status.setText("导入失败：" + resp));
            }
        } catch (Exception e) {
            importing = false;
            Log.e(TAG, "import failed", e);
            ui.post(() -> status.setText("导入出错：" + e.getMessage()));
        } finally {
            if (c != null) c.disconnect();
        }
    }

    @Override
    protected void onDestroy() {
        if (poller != null) ui.removeCallbacks(poller);
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }
}
