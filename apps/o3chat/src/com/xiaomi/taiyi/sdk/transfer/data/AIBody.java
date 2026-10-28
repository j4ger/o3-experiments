package com.xiaomi.taiyi.sdk.transfer.data;

import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;

/**
 * Client-side stub for AIService's AIBody.
 *
 * <p>The real {@code AIBody} extends {@code AIPackage}, whose
 * {@code writeToParcel} writes one {@link Bundle}; {@code AIBody} writes a
 * second bundle for its frame map.  The server reads both.</p>
 */
public final class AIBody implements Parcelable {
    public static final String KEY_JSON_SOURCE = "json_source";

    public Bundle bundle = new Bundle();
    public Bundle frames = new Bundle();

    public AIBody() {
    }

    private AIBody(Parcel in) {
        readFromParcel(in);
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeBundle(bundle);
        dest.writeBundle(frames);
    }

    public void readFromParcel(Parcel in) {
        Bundle b = in.readBundle(AIBody.class.getClassLoader());
        bundle = b != null ? b : new Bundle();
        bundle.setClassLoader(AIBody.class.getClassLoader());

        Bundle f = in.readBundle(AIBody.class.getClassLoader());
        frames = f != null ? f : new Bundle();
        frames.setClassLoader(AIBody.class.getClassLoader());
    }

    /** Mirrors the real AIBody.setJsonSource(). */
    public void setJsonSource(String json) {
        bundle.putString(KEY_JSON_SOURCE, json);
    }

    /** Mirrors the real AIBody.getJsonSource(). */
    public String getJsonSource() {
        return bundle.getString(KEY_JSON_SOURCE, "");
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Parcelable.Creator<AIBody> CREATOR =
            new Parcelable.Creator<AIBody>() {
                @Override
                public AIBody createFromParcel(Parcel in) {
                    return new AIBody(in);
                }

                @Override
                public AIBody[] newArray(int size) {
                    return new AIBody[size];
                }
            };
}
