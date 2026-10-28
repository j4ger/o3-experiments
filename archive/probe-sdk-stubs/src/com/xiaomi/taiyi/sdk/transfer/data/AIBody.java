package com.xiaomi.taiyi.sdk.transfer.data;

import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;

/**
 * Client-side stub for AIService's AIBody.  AIPackage writes one Bundle and
 * AIBody writes a second Bundle for frames.
 */
public final class AIBody implements Parcelable {
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
